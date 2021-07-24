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
start = time.time()

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

net = Net(20)
if train:
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()
else:
    net.load_state_dict(torch.load("models/value_function.dict"))

if len(sys.argv)>1:
    initial_drivers = int(sys.argv[1])
else:
    initial_drivers = 10
A_coeff = 10
M_coeff = 1
delta = 3
epsilon = 10
GROUPS = 10
gamma = 0.5

drivers = get_initial_drivers(initial_drivers)
k,k_matrix = get_groups(GROUPS)
data = Data()

for epoch in range(TOTAL_EPOCHS):
    if epoch % 60 == 0:
            print(epoch//60)
            print("Total Profit {}".format(round(data.total_profit)))

    riders = read_riders(A_coeff,M_coeff)
    update_drivers(drivers,epoch)
    rider_costs = [get_ride_cost(i,A_coeff) for i in riders]
    rider_valuation = [get_valuation(i,M_coeff,k) for i in riders]

    m = len(riders)
    n = len(drivers)

    baseline_state = get_current_state(drivers,epoch)+k
    
    price_pairs = {}
    objective_values = {}

    net_data = []
    targets = []

    for i in range(m):
        for j in range(n):
            cost = rider_costs[i]

            current_state = deepcopy(baseline_state)+[regions[riders[i].start],riders[i].group]
            best_pair = {'value':gamma*value_function(current_state),'price':0}

            if not drivers[j].occupied:
                for price in np.arange(cost,rider_valuation[i],.1):
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

    matches = solve_model(objective_values,m,n,riders,drivers,delta)
        
    prices = {}
    costs = {} 
    valuations = {}

    for (i,j) in matches:
        current_state = deepcopy(baseline_state)+[regions[riders[i].start],riders[i].group]
        prices[i] = price_pairs[(i,j)]
        costs[i] = rider_costs[i]
        valuations[i] = rider_valuation[i]
        if train:
            targets.append([objective_values[(i,j)]])
            net_data.append(current_state)

    

    k_addition = new_k(matches,valuations,prices,k_matrix,riders)

    for i in range(len(k)):
        k[i]+=k_addition[i]

    update_data(data,prices,costs,valuations,riders,drivers)
    move_drivers(riders,drivers,matches,epoch)

    if len(net_data)>0:
        optimizer.zero_grad()   
        output = net(torch.tensor(np.array(net_data)).float())
        loss = criterion(output, torch.tensor(np.array(targets)).float())
        loss.backward()
        optimizer.step()

if train:
    torch.save(net.state_dict(),"models/value_function.dict")

print("Total profit {}".format(data.total_profit))
data_dict = data.__dict__()
data_dict['type'] = 'neural'
data_dict['A_coeff'] = A_coeff
data_dict['M_coeff'] = M_coeff
data_dict['delta'] = delta
data_dict['epsilon'] = epsilon
data_dict['initial_drivers'] = initial_drivers
data_dict['GROUPS'] = GROUPS
data_dict['gamma'] = gamma
data_dict['time'] = time.time()-start
pickle.dump(data_dict,open("results/"+str(int(time.time()))+".p","wb"))
