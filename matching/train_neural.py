from util import *
from copy import deepcopy
from docplex.mp.model import Model  # type: ignore
from docplex.mp.linear import Var  # type: ignore
import numpy as np
import time 

A_coeff = 10
M_coeff = 1
delta = 3
epsilon = 10
initial_drivers = 10
GROUPS = 10

num_regions = 5
regions = get_k_regions(5)

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
    
    for i in range(m):
        for j in range(n):
            variable = model.continuous_var(name='x{}_{}'.format(i, j))
            names.append(variable)
            objective.append(rider_valuation[i]-rider_costs[i])
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
    current_state = num_by_location+[idle_drivers,busy_drivers]+k

    for i in range(m):
        for j in range(n):
            if solution.get_value("x{}_{}".format(i,j)) == 1:
                matches.append((i,j))
                g = riders[i].group
                cost = rider_costs[i]

                for price in range(100):
                    new_state = deepcopy(current_state)
                    # Calculate value function

                    print(cost,price,rider_valuation[i])
                    
                    if price>cost and price<=rider_valuation[i]:
                        reward = price-cost
                        new_state[regions[drivers[j].location]]-=1
                        new_state[regions[riders[i].end]]+=1
                        for group_num in range(GROUPS):
                            new_state[len(num_by_location)+2+group_num]+=k_matrix[group_num][g]*(rider_valuation[i]-price)
                        new_state[len(num_by_location)]-=1
                        new_state[len(num_by_location)+1]+=1
                    else:
                        reward = 0
                    new_state+=[regions[riders[i].start],g]

                    if reward>0:
                        print("Reward {}, New State {}".format(reward,new_state))
                
                price = price_baseline(riders[i])

                for group_num in range(GROUPS):
                    k_addition[group_num]+=k_matrix[group_num][g]*(rider_valuation[i]-price)
                
                rider = riders[i]
                dist = travel_times[rider.start][rider.end]
                time_to_get = travel_times[drivers[j].location][rider.start]
                drivers[j].set_occupied(True,time_to_get+dist+epoch)
                drivers[j].location = rider.end

    for i in range(GROUPS):
        k[i]+=k_addition[i]

