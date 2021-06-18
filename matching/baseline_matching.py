import random
import numpy as np
import time
from math import e
from docplex.mp.model import Model  # type: ignore
from docplex.mp.linear import Var  # type: ignore
import matplotlib.pyplot as plt

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

driver_locations = open("data/taxi_3000_final.txt").read().strip().split("\n")
driver_locations = [int(i) for i in driver_locations]

time_start = time.time()

ride_data = open("data/test_flow_5000_1.txt")
TOTAL_EPOCHS = int(ride_data.readline())

time_per_match = 60
travel_times = np.load('data/zone_traveltime.npy')
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
profit_over_time = []
demand_over_time = []
unsatisfied_demand_over_time = []

num_occupied_drivers = []

for epoch in range(TOTAL_EPOCHS):
    if epoch % 60 == 0:
        print(epoch//60)
    
    time_start = time.time()
    line = ride_data.readline()
    rider_data = []
    while "Flows" not in line and line!='':
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
    
    for i in range(m):
        for j in range(n):
            if solution.get_value("x{}_{}".format(i,j)) == 1:
                matches.append((i,j))
                price = rider_valuation[i]
                cost = rider_costs[i]
                total_profit+=(price-cost)

                g = riders[i].group
                for group_num in range(GROUPS):
                    k_addition[group_num]+=k_matrix[group_num][g]*(rider_valuation[i]-price)
                
                rider = riders[i]
                dist = travel_times[rider.start][rider.end]
                time_to_get = travel_times[drivers[j].location][rider.start]
                drivers[j].set_occupied(True,time_to_get+dist+epoch)

    num_occupied_drivers.append(len([i for i in drivers if i.occupied]))

    for i in range(GROUPS):
        k[i]+=k_addition[i]
                
    profit_over_time.append(total_profit)
    demand_over_time.append(m)
    unsatisfied_demand_over_time.append(m-len(matches))

profit_over_time = np.array(profit_over_time)
demand_over_time = np.array(demand_over_time)
unsatisfied_demand_over_time = np.array(unsatisfied_demand_over_time)

plt.figure(figsize=(10,10))
plt.title("Profit over time",fontsize=20)
plt.xlabel("Epoch",fontsize=14)
plt.ylabel("Profit",fontsize=14)
plt.xticks([0,8*60,16*60,24*60],fontsize=14)
plt.yticks(fontsize=14)
plt.plot(profit_over_time)
plt.savefig("images/profit_baseline.png")
#plt.show()

plt.figure(figsize=(10,10))
plt.title("Demand over time",fontsize=20)
plt.xlabel("Epoch",fontsize=14)
plt.ylabel("Demand",fontsize=14)
plt.xticks([0,8*60,16*60,24*60],fontsize=14)
plt.yticks(fontsize=14)
plt.plot(demand_over_time)
plt.savefig("images/demand_baseline.png")
#plt.show()

plt.figure(figsize=(10,10))
plt.title("Unsatisfied demand over time",fontsize=20)
plt.xlabel("Epoch",fontsize=14)
plt.ylabel("Unsatisfied Demand",fontsize=14)
plt.xticks([0,8*60,16*60,24*60],fontsize=14)
plt.yticks(fontsize=14)
plt.plot(unsatisfied_demand_over_time)
plt.savefig("images/no_demand_baseline.png")
#plt.show()

print("Total Profit {}".format(total_profit))
