import sys
from util import *
from copy import deepcopy
from docplex.mp.model import Model  # type: ignore
from docplex.mp.linear import Var  # type: ignore
import numpy as np
import time
import torch
import torch.optim as optim
import pickle
import time
import matplotlib.pyplot as plt

def objective_function(price,cost):
    return -(price-cost)

start = time.time()

# Load in settings
file_name = "settings/model_settings.txt"
if len(sys.argv)>1:
    file_name = sys.argv[1]
settings_list = read_from_file(file_name)

# How much should the rider get paid per hour? What's their value
rider_valuation_of_driver = settings_list['driver_opportunity_cost_avg']
# How much should the rider pay?
# This is the amount that the firm (uber) is worth + how much the driver is worth
# Frictional multiplier scales because of down time (in an hour, only 30 minutes is spent driving)
rider_avg_price_per_hour = (settings_list['rider_valuation_of_firm']
                                             + rider_valuation_of_driver)*settings_list['frictional_multiplier']

# What percent of the money should drivers receive, and how much goes to the firm?
driver_comission = (rider_valuation_of_driver)/(settings_list['rider_valuation_of_firm']
                                                                               + rider_valuation_of_driver)

# Create a network for the value function
net = Net(20)
if settings_list['train']:
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()
else:
    net.load_state_dict(torch.load("models/{}".format(settings_list['model_name'])))

# Maintain driver state + externality numbers
all_drivers = get_initial_drivers(settings_list['total_drivers'],settings_list['driver_opportunity_cost_avg'])
drivers = all_drivers[:settings_list['initial_drivers']]
k,k_matrix = get_groups(settings_list['GROUPS'])
data = Data()

# If we're training, use first 10 days, otherwise last 5
which_days = range(1,11)
if not settings_list['train']:
    which_days = range(11,16)

print("Setup driver data")

for day in which_days[:settings_list['num_days']]:
    print("On day {}".format(day))
    reset_day(day)
    
    for epoch in range(settings_list['TOTAL_EPOCHS']):
        # Debugging 
        if epoch%60 == 0 and epoch>0:
                print("Hour number {}".format(epoch//60))
                print("Total Profit {}".format(round(data.total_profit)))
                print("Number of drivers {}".format(len(drivers)))

        if settings_list['real_riders']:
            # Get riders based on New York Data
            riders = read_riders(rider_avg_price_per_hour,settings_list['GROUPS'])
        else:
            # Get num_rides randomly, each centered around rider_avg_price
            riders = random_rides(settings_list['num_rides'],rider_avg_price_per_hour,settings_list['GROUPS'])

        # Based on driving history, update as drivers enter/leave the system
        drivers = update_drivers(drivers,all_drivers,epoch,data,driver_comission)

        # How much are riders willing to pay, based on their valuation + externalities 
        rider_valuation = [get_valuation(i,settings_list['externality_multiplier'],k) for i in riders]

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
                # What is the optimal price for rider i, driver j? 
                cost = 0.5*rider_valuation[i]
                current_state = deepcopy(baseline_state)+[regions[riders[i].start],riders[i].group]

                # The best price is 0; not being serviced 
                best_pair = {'value':settings_list['gamma']*value_function(net,current_state),'price':0}

                # If the driver can service, try 100 different price ranges
                # From cost (the most we can do to break even) to 150% more
                if not drivers[j].occupied:
                    if rider_valuation[i] == 0:
                        price = 0
                        prie_range = [0]
                    else:
                        price_range = np.arange(cost,rider_valuation[i]*1.5,(rider_valuation[i]*1.5-cost)/100)

                    # Compute the immideate reward, and the long term value
                    for price in price_range:
                        new_state = deepcopy(current_state)

                        # OBJ: Change reward
                        reward = objective_function(price,cost)
                        new_state = update_state(new_state,riders[i],drivers[j],k,k_matrix,rider_valuation[i],price,epoch)
                        total_value = reward + settings_list['gamma']*value_function(net,new_state)

                        if total_value>best_pair['value']:
                            best_pair = {'value': total_value, 'price': price}

                # If the best price is 0, don't match them
                if best_pair['price'] == 0:
                    price_pairs[(i,j)] = 0
                    objective_values[(i,j)] = -100000
                else:
                    price_pairs[(i,j)] = best_pair['price']
                    objective_values[(i,j)] = best_pair['value']

        # Solve the LP, based on MDP values
        matches = solve_model(objective_values,m,n,riders,drivers,settings_list['delta'])
            
        prices = {}
        costs = {} 
        valuations = {}

        driver_extra_pay = {}
        if epoch>60:
            future_demand = predict_future_demand(data,epoch)
            current_demand = m
        
        # Get matching LP
        for (i,j) in matches:
            current_state = deepcopy(baseline_state)+[regions[riders[i].start],riders[i].group]
            prices[i] = price_pairs[(i,j)]
            costs[i] = 0.5*rider_valuation[i]

            # Should we pay the driver extra, at a cost to the firm? 
            driver_extra_pay[i] = 0
            if epoch>60:
                if future_demand>current_demand*settings_list['extra_pay_cutoff']:
                    driver_extra_pay[i] = settings_list['extra_pay_multiplier']*prices[i]

            # Update the network, so it associates our current state with an objective function
            valuations[i] = rider_valuation[i]
            if settings_list['train']:
                targets.append([objective_values[(i,j)]])
                net_data.append(current_state)

        # Update K based on matches
        k_addition = new_k(matches,valuations,prices,k_matrix,riders)
        for i in range(len(k)):
            k[i]+=k_addition[i]


        # Update data we track + move drivers
        update_data(data,prices,costs,driver_extra_pay,valuations,riders,drivers)
        move_drivers(riders,drivers,matches,epoch)

        # If we're training, run loss on the model
        if settings_list['train'] and len(matches)>0:
            optimizer.zero_grad()   
            output = net(torch.tensor(np.array(net_data)).float())
            loss = criterion(output, torch.tensor(np.array(targets)).float())
            loss.backward()
            optimizer.step()

    print("Finished one day")

if settings_list['train']:
    torch.save(net.state_dict(),"models/value_function_{}.dict".format(settings_list['num_days']))

print("Total profit {}".format(data.total_profit))
data_dict = data.__dict__()

for i in settings_list:
    data_dict[i] = settings_list[i]

data_dict['time'] = time.time()-start
data_dict['k'] = k
pickle.dump(data_dict,open("results/"+str(int(time.time()))+".p","wb"))
