import random
import numpy as np
import time
from scipy.optimize import linprog
from math import e

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

print("Driver locations took {} time".format(time.time()-time_start))
time_start = time.time()

ride_data = open("test_flow_5000_1.txt")
TOTAL_EPOCHS = int(ride_data.readline())
print("Rider data took {} time".format(time.time()-time_start))

time_per_match = 60
travel_times = np.load('zone_traveltime.npy')
travel_times = np.round(travel_times/time_per_match)

print("Travel times took {} time".format(time.time()-time_start))
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

print("Setup took {} time".format(time.time()-time_start))

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

    print("Update drivers/riders took {} time".format(time.time()-time_start))
    time_start = time.time()
    
    rider_costs = [A_coeff*travel_times[i.start][i.end] for i in riders]
    rider_valuation = [i.value+M_coeff*sigmoid(k[i.group]) for i in riders]

    m = len(riders)
    n = len(drivers)

    # Setup the LP
    objective = []
    
    for i in range(m):
        for j in range(n):
            objective.append(rider_costs[i]-rider_valuation[i])

    A = []
    b = []            

    not_allowed = set()
    count = 0
    for i in range(m):
        for j in range(n):
            rider_loc = riders[i].start
            driver_loc = drivers[j].location

            if travel_times[driver_loc][rider_loc]>delta:
                not_allowed.add(count)
                temp = [0 for i in range(n*m)]
                temp[count] = 1
                A.append(temp)
                b.append(0)

            count+=1

        
    for i in range(m):
        temp = []
        for i2 in range(m):
            for j in range(n):
                if i2 == i and i2*n+j not in not_allowed:
                    temp.append(1)
                else:
                    temp.append(0)
        A.append(temp)
        b.append(1)

    for j in range(n):
        temp = []
        for i in range(m):
            for j2 in range(n):
                if j2 == j and i*n+j2 not in not_allowed:
                    temp.append(1)
                else:
                    temp.append(0)
        A.append(temp)
        b.append(int(not drivers[j].occupied))
    
    bnd = [(0,1) for i in range(n*m)]
    print("LP setup time {}".format(time.time()-time_start))
    time_start = time.time()

    opt = linprog(c=objective, A_ub=A, b_ub=b, bounds=bnd)
    print("LP solution time {}".format(time.time()-time_start))
    print("{} variables, {} constriants".format(n*m,len(A)))
    time_start = time.time()
    
    matches = []
    
    for i in range(m):
        for j in range(n):
            if opt.x[i*n+j] == 1:
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
                
    print("Post LP took {} time".format(time.time()-time_start))
    break
