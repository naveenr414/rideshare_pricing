import random
import numpy as np
import time
from math import e
import cplex

class Driver:
    def __init__(self,time_in,time_out,initial_location):
        self.time_in = time_in
        self.time_out = time_out
        self.initial_location = initial_location
        self.occupied = False
        self.free_epoch = -1
        self.location = initial_location

    def set_occupied(self,occupied,free_epoch):
        self.occupied = occupied
        self.free_epoch = free_epoch

class Rider:
    def __init__(self,start,end,value,group):
        self.start = start
        self.end = end
        self.value = value
        self.group = group

def sigmoid(x):
    return  1/(1+e**(-x))

time_start = time.time()

driver_locations = open("taxi_3000_final.txt").read().strip().split("\n")
driver_locations = [int(i) for i in driver_locations]

time_start = time.time()

ride_data = open("test_flow_5000_1.txt")
TOTAL_EPOCHS = int(ride_data.readline())

time_per_match = 60
travel_times = np.load('zone_traveltime.npy')
travel_times = np.round(travel_times/time_per_match)

time_start = time.time()

A_coeff = 1
M_coeff = 1
delta = 3
epsilon = 10
initial_drivers = 10

ride_data.readline()

drivers = [Driver(0,TOTAL_EPOCHS,i) for i in driver_locations[:initial_drivers]]

GROUPS = 10
k = [0 for i in range(GROUPS)]
k_matrix = [[random.random()/10 for i in range(GROUPS)] for j in range(GROUPS)]
for i in range(GROUPS):
    k_matrix[i][i] = random.random()/10+0.2

total_profit = 0

for epoch in range(TOTAL_EPOCHS):
    time_start = time.time()
    line = ride_data.readline()
    rider_data = []
    while "Flows" not in line:
        rider_data.append(line)
        line = ride_data.readline()

    riders = []
    for i in range(len(rider_data)):
        temp = rider_data[i].split(",")
        start = int(temp[0])
        end = int(temp[1])
        valuation = A_coeff*travel_times[start][end]+(random.random()-0.5)*A_coeff/4
        riders.append(Rider(start,end,valuation,random.randint(0,GROUPS-1)))

    for i in range(len(drivers)):
        if drivers[i].occupied and drivers[i].free_epoch<=epoch:
            drivers[i].set_occupied(False,-1)

    time_start = time.time()
    
    rider_costs = [A_coeff*travel_times[i.start][i.end] for i in riders]
    rider_valuation = [i.value+M_coeff*sigmoid(k[i.group]) for i in riders]

    m = len(riders)
    n = len(drivers)

    problem = cplex.Cplex()
    problem.set_log_stream(None)
    problem.set_error_stream(None)
    problem.set_warning_stream(None)
    problem.set_results_stream(None)


    problem.objective.set_sense(problem.objective.sense.maximize)
    names = []
    objective = []
    
    for i in range(m):
        for j in range(n):
            names.append("x{}_{}".format(i,j))
            objective.append(rider_valuation[i]-rider_costs[i])

    lower_bounds = [0 for i in range(n*m)]
    upper_bounds = []
    for i in range(m):
        for j in range(n):
            driver_loc = drivers[j].location
            rider_loc = riders[i].start
            if drivers[j].occupied or travel_times[driver_loc][rider_loc]>delta:
                upper_bounds.append(0)
            else:
                upper_bounds.append(1)
    problem.variables.add(obj = objective,
                          lb = lower_bounds,
                          ub = upper_bounds,
                          names = names)


    constraint_names = []
    constraints = []

    for i in range(m):
        constraint_names.append("c{}".format(i))
        variables = []
        for j in range(n):
            variables.append("x{}_{}".format(i,j))
        constraints.append([variables,[1 for i in range(len(variables))]])


    for j in range(n):
        constraint_names.append("d{}".format(j))
        variables = []
        for i in range(m):
            variables.append("x{}_{}".format(i,j))

        constraints.append([variables,[1 for i in range(len(variables))]])
    rhs = [1 for i in range(len(constraints))]
    constraint_senses = ["L" for i in range(len(rhs))]
    
    time_start = time.time()

    problem.linear_constraints.add(lin_expr = constraints,
                               senses = constraint_senses,
                               rhs = rhs,
                               names = constraint_names)
    problem.solve()

    time_start = time.time()
    
    matches = []
    x = problem.solution.get_values()
    
    for i in range(m):
        for j in range(n):
            if x[i*n+j] == 1:
                matches.append((i,j))
                price = rider_valuation[i]
                cost = rider_costs[i]
                total_profit+=(price-cost)

                g = riders[i].group
                for group_num in range(GROUPS):
                    k[group_num]+=k_matrix[group_num][g]*(rider_valuation[i]-price)
                
                rider = riders[i]
                dist = travel_times[rider.start][rider.end]
                time_to_get = travel_times[drivers[j].location][rider.start]
                drivers[j].set_occupied(True,time_to_get+dist+epoch)
                
    print(total_profit)
