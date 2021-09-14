from util import *
from copy import deepcopy
from docplex.mp.model import Model  # type: ignore
from docplex.mp.linear import Var  # type: ignore
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import sys
import time

train = False

class Net(nn.Module):

    def __init__(self,input_size):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.fc1 = nn.Linear(input_size, 32)  # 5*5 from image dimension
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def value_function(state):
    return float(net(torch.tensor(np.array(state)).float())[0])

def price_baseline(rider):
    return rider.value+M_coeff*sigmoid(k[rider.group])


start = time.time()
net = Net(20)
if train:
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()
else:
    net.load_state_dict(torch.load("models/value_function.dict"))

# PARAMETERS
frictional_multiplier = 1
externality_multiplier = 2 
if len(sys.argv)>1:
    initial_drivers = int(sys.argv[1])
else:
    initial_drivers = 1
total_drivers = initial_drivers*1
rider_valuation_of_firm = 10
driver_opportunity_cost_avg = 40
rider_valuation_of_driver = driver_opportunity_cost_avg
rider_avg_price_per_hour = (rider_valuation_of_firm + rider_valuation_of_driver)*frictional_multiplier
driver_comission = (rider_valuation_of_driver)/(rider_valuation_of_firm + rider_valuation_of_driver)
delta = 3
epsilon = 10
GROUPS = 10
gamma = 0.5
num_rides = 20
real_riders = True

print("Setup parameters")

# Get Driver Data
all_drivers = get_initial_drivers(total_drivers,driver_opportunity_cost_avg)
drivers = all_drivers[:initial_drivers]
k,k_matrix = get_groups(GROUPS)
data = Data()
which_days = range(1,11)
if not train:
    which_days = range(11,16)

print("Setup driver data")

for day in which_days:
    print("On day {}".format(day))
    reset_day(day)
    for epoch in range(TOTAL_EPOCHS):

        # Debugging 
        if epoch%60 == 0 and epoch>0:
                print("Hour number {}".format(epoch//60))
                print("Total Profit {}".format(round(data.total_profit)))

                services = data.serviced_riders_over_time[-60:]
                revenue_over_time = data.revenue_over_time[-60:]
                profit_over_time = data.profit_over_time[-60:]

                if np.sum(services)!=0:
                    print("Profit/service {}".format(np.sum(profit_over_time)/np.sum(services)))
                    print("Revenue/service {}".format(np.sum(revenue_over_time)/np.sum(services)))
                print("There are {} drivers".format(len(drivers)))

        if real_riders:
            # Get riders based on New York Data
            riders = read_riders(rider_avg_price_per_hour,GROUPS)
        else:
            # Get num_rides randomly, each centered around rider_avg_price
            riders = random_rides(num_rides,rider_avg_price_per_hour,GROUPS)

        # Based on driving history, update how many drivers enter/leave the system
        drivers = update_drivers(drivers,all_drivers,epoch,data,driver_comission)

        # Now max revenue 
        rider_valuation = [get_valuation(i,externality_multiplier,k) for i in riders]

        m = len(riders)
        n = len(drivers)

        # Setup the LP
        baseline_state = get_current_state(drivers,epoch)+k
        
        price_pairs = {}
        objective_values = {}

        net_data = []
        targets = []

        for i in range(m):
            for j in range(n):
                cost = 0.9*rider_valuation[i]

                current_state = deepcopy(baseline_state)+[regions[riders[i].start],riders[i].group]
                best_pair = {'value':gamma*value_function(current_state),'price':0}

                if not drivers[j].occupied:
                    if rider_valuation[i] == cost == 0:
                        price = 0
                        prie_range = [0]
                    else:
                        price_range = np.arange(cost,rider_valuation[i],(rider_valuation[i]-cost)/100)
                    
                    for price in price_range:
                        new_state = deepcopy(current_state)
                        reward = price-cost
                        new_state = update_state(new_state,riders[i],drivers[j],k,k_matrix,rider_valuation[i],price,epoch)
                        total_value = reward + gamma*value_function(new_state)

                        if total_value>best_pair['value']:
                            best_pair = {'value': total_value, 'price': price}

                if best_pair['price'] == 0:
                    price_pairs[(i,j)] = 0
                    objective_values[(i,j)] = -100000
                else:
                    price_pairs[(i,j)] = best_pair['price']
                    objective_values[(i,j)] = best_pair['value']

        # Solve the LP, based on MDP values
        matches = solve_model(objective_values,m,n,riders,drivers,delta)
            
        prices = {}
        costs = {} 
        valuations = {}

        # Get matching LP
        for (i,j) in matches:
            current_state = deepcopy(baseline_state)+[regions[riders[i].start],riders[i].group]
            prices[i] = price_pairs[(i,j)]
            costs[i] = 0 
            valuations[i] = rider_valuation[i]
            if train:
                targets.append([objective_values[(i,j)]])
                net_data.append(current_state)

        # Update K based on matches
        k_addition = new_k(matches,valuations,prices,k_matrix,riders)
        for i in range(len(k)):
            k[i]+=k_addition[i]

        # Update data we track + move drivers
        update_data(data,prices,costs,valuations,riders,drivers)
        move_drivers(riders,drivers,matches,epoch)

        # If we're training, run loss on the model
        if len(net_data)>0:
            optimizer.zero_grad()   
            output = net(torch.tensor(np.array(net_data)).float())
            loss = criterion(output, torch.tensor(np.array(targets)).float())
            loss.backward()
            optimizer.step()
    print("Finished one day")

if train:
    torch.save(net.state_dict(),"models/value_function.dict")

print("Total profit {}".format(data.total_profit))
data_dict = data.__dict__()
data_dict['type'] = 'neural'
data_dict['rider_valuation_of_firm'] = rider_valuation_of_firm
data_dict['driver_opportunity_cost_avg'] = driver_opportunity_cost_avg
data_dict['externality_multiplier'] = externality_multiplier
data_dict['frictional_multiplier'] = frictional_multiplier
data_dict['num_rides'] = num_rides
data_dict['delta'] = delta
data_dict['epsilon'] = epsilon
data_dict['initial_drivers'] = initial_drivers
data_dict['GROUPS'] = GROUPS
data_dict['gamma'] = gamma
data_dict['time'] = time.time()-start
data_dict['k'] = k
pickle.dump(data_dict,open("results/"+str(int(time.time()))+".p","wb"))
