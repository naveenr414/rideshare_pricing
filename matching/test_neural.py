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

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.fc1 = nn.Linear(19, 32)  # 5*5 from image dimension
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def value_function(state):
    return float(net(torch.tensor(np.array(state)).float())[0])

net = Net()
net.load_state_dict(torch.load("value_function.dict"))

A_coeff = 10
M_coeff = 1
delta = 3
epsilon = 10
initial_drivers = 10
GROUPS = 10
gamma = 0.9

num_regions = 5
regions = get_k_regions(5)
total_profit = 0

def price_baseline(rider):
    return rider.value+M_coeff*sigmoid(k[rider.group])

drivers = get_initial_drivers(initial_drivers) 
k = [0 for i in range(GROUPS)]
k_matrix = [[random.random()/10 for i in range(GROUPS)] for j in range(GROUPS)]
for i in range(GROUPS):
    k_matrix[i][i] = random.random()/10+0.2

for epoch in range(TOTAL_EPOCHS):
    if epoch % 60 == 0:
            print(epoch//60)

    riders = read_riders(A_coeff,M_coeff,GROUPS)
    for i in range(len(drivers)):
        if drivers[i].occupied and drivers[i].free_epoch<=epoch:
            drivers[i].set_occupied(False,-1)
    rider_costs = [A_coeff*travel_times[i.start][i.end] for i in riders]
    rider_valuation = [i.value+M_coeff*sigmoid(k[i.group]) for i in riders]

    m = len(riders)
    n = len(drivers)

    model = Model()
    names = []
    objective = []

    current_state = []
    num_by_location = [0 for i in range(num_regions)]

    idle_drivers = 0
    busy_drivers = 0

    for i in range(len(drivers)):
        num_by_location[regions[drivers[i].location]]+=1
        if drivers[i].occupied:
            busy_drivers+=1
        else:
            idle_drivers+=1

    optimal_price_rider_driver = []
    objective_values = []
    for i in range(m):
        temp = []
        temp2 = []
        for j in range(n):
            current_state = num_by_location+[idle_drivers,busy_drivers]+k
            g = riders[i].group
            cost = rider_costs[i]
            new_state = deepcopy(current_state)
            new_state+=[regions[riders[i].start],g]
            # No action reward
            best_value = 0+value_function(new_state)
            best_price = 0

            if not drivers[j].occupied:
                for price in np.arange(cost,rider_valuation[i],.1):
                    new_state = deepcopy(current_state)
                    
                    temp_reward = price-cost
                    new_state[regions[drivers[j].location]]-=1
                    new_state[regions[riders[i].end]]+=1
                    for group_num in range(GROUPS):
                        new_state[len(num_by_location)+2+group_num]+=k_matrix[group_num][g]*(rider_valuation[i]-price)
                    new_state[len(num_by_location)]-=1
                    new_state[len(num_by_location)+1]+=1
                    new_state+=[regions[riders[i].start],g]

                    temp_value = temp_reward + gamma*value_function(new_state)

                    if temp_value>best_value:
                        best_value = temp_value
                        best_price = price
            temp.append(best_price)
            temp2.append(best_value)
        optimal_price_rider_driver.append(temp)
        objective_values.append(temp2)
        
    for i in range(m):
        for j in range(n):
            variable = model.continuous_var(name='x{}_{}'.format(i, j))
            names.append(variable)
            objective.append(objective_values[i][j])
            driver_loc = drivers[j].location
            rider_loc = riders[i].start
            upper_bound = int(not(drivers[j].occupied or travel_times[driver_loc][rider_loc]>delta))
            
            model.add_constraint(variable<=upper_bound)
    score = model.sum(objective[i] * names[i] for i in range(len(names)))
    model.maximize(score)



    for i in range(m):
        variables = []
        for j in range(n):
            variables.append(i*n+j)
        model.add_constraint(model.sum(names[o] for o in variables)<=1)


    for j in range(n):
        variables = []
        for i in range(m):
            variables.append(i*n+j)
        model.add_constraint(model.sum(names[o] for o in variables)<=1)
    
    time_start = time.time()
    solution = model.solve()

    
    matches = []

    k_addition = [0 for i in range(GROUPS)]



    for i in range(m):
        for j in range(n):
            if solution.get_value("x{}_{}".format(i,j)) == 1:
                matches.append((i,j))
                g = riders[i].group
                cost = rider_costs[i]
                                
                price = optimal_price_rider_driver[i][j]

                if price<=rider_valuation[i]:
                    total_profit+=price
                    for group_num in range(GROUPS):
                        k_addition[group_num]+=k_matrix[group_num][g]*(rider_valuation[i]-price)
                    
                    rider = riders[i]
                    dist = travel_times[rider.start][rider.end]
                    time_to_get = travel_times[drivers[j].location][rider.start]
                    drivers[j].set_occupied(True,time_to_get+dist+epoch)
                    drivers[j].location = rider.end

    for i in range(GROUPS):
        k[i]+=k_addition[i]
    
print("Total profit {}".format(total_profit))
