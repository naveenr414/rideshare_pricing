import random
import numpy as np
import time
from util import *

def price_baseline(rider):
    return rider.value+M_coeff*sigmoid(k[rider.group])

def price_maximal(rider):
    return epsilon + rider.value + M_coeff*sigmoid(k[rider.group])

A_coeff = 10
M_coeff = 1
delta = 3
epsilon = 10
initial_drivers = 10
GROUPS = 10

drivers = get_initial_drivers(initial_drivers)
k,k_matrix = get_groups(GROUPS)
data = Data()

for epoch in range(TOTAL_EPOCHS):
    if epoch % 60 == 0:
        print(epoch//60)
        print("Total Profit {}".format(data.total_profit))

     
    riders = read_riders(A_coeff,GROUPS)
    update_drivers(drivers,epoch)

    rider_costs = [get_ride_cost(i,A_coeff) for i in riders]
    rider_valuation = [get_valuation(i,M_coeff,k) for i in riders]

    m = len(riders)
    n = len(drivers)


    objective_values = {}
    for i in range(m):
        for j in range(n):
            objective_values[(i,j)] = rider_valuation[i]-rider_costs[i]

    matches = solve_model(objective_values,m,n,riders,drivers,delta)

    prices = {}
    costs = {} 
    valuations = {}

    for (i,j) in matches:
        prices[i] = price_baseline(riders[i])
        costs[i] = rider_costs[i]
        valuations[i] = rider_valuation[i]
    k_addition = new_k(matches,valuations,prices,k_matrix,riders)

    for i in range(len(k)):
        k[i]+=k_addition[i]

    update_data(data,prices,costs,valuations,riders,drivers)
    move_drivers(riders,drivers,matches,epoch)

print("Total Profit {}".format(data.total_profit))
