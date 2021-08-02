import random
import numpy as np
from math import e
from sklearn.cluster import KMeans
from docplex.mp.model import Model  # type: ignore
from docplex.mp.linear import Var  # type: ignore
import matplotlib.pyplot as plt

class Data:
    def __init__(self):
        self.total_profit = 0
        self.total_profit_over_time = []
        self.profit_over_time = []
        self.demand_over_time = []
        self.unsatisfied_demand_over_time = []
        self.num_occupied_drivers = []
        self.revenue_over_time = []
        self.serviced_riders_over_time = []
        self.num_drivers_over_time = []

    def __dict__(self):
        return {'total_profit': self.total_profit,
                'profit_over_time': self.profit_over_time,
                'demand_over_time': self.demand_over_time,
                'unsatisfied_demand_over_time': self.unsatisfied_demand_over_time,
                'num_occupied_drivers': self.num_occupied_drivers,
                'serviced_riders_over_time': self.serviced_riders_over_time,
                'revenue_over_time': self.revenue_over_time,
                'total_profit_over_time': self.total_profit_over_time,
                'num_drivers_over_time': self.num_drivers_over_time}

class Driver:
    def __init__(self,time_in,time_out,initial_location):
        self.time_in = time_in
        self.time_out = time_out
        self.initial_location = initial_location
        self.occupied = False
        self.free_epoch = -1
        self.location = initial_location
        self.min_price = np.random.normal(1.4,0.5)

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
    return  2/(1+e**(-x))-1
def get_k_regions(num_regions):    
    a = open("data/zone_latlong.csv").read().strip().split("\n")    
    a = [[i.split(",")[1],i.split(",")[2]] for i in a]
    a = [[float(i[0]),float(i[1])] for i in a]
    a = np.array(a)

    kmeans = KMeans(n_clusters=num_regions,random_state=0).fit(a)
    return kmeans.labels_

driver_locations = open("data/taxi_3000_final.txt").read().strip().split("\n")
driver_locations = [int(i) for i in driver_locations]

ride_data = open("data/test_flow_5000_1.txt")
TOTAL_EPOCHS = int(ride_data.readline())
ride_data.readline()

time_per_match = 60
travel_times = np.load('data/zone_traveltime.npy')
travel_times = np.round(travel_times/time_per_match)

num_regions = 5
regions = get_k_regions(num_regions)

def get_ride_cost(rider,A_coeff):
    return A_coeff*travel_times[rider.start][rider.end]

def get_valuation(rider,M_coeff,k):
    return rider.value+M_coeff*sigmoid(k[rider.group])

def read_riders(A_coeff,GROUPS):
    rider_data = []
    line = ride_data.readline()
    while "Flows" not in line and line!='':
        rider_data.append(line)
        line = ride_data.readline()

    riders = []
    for i in range(len(rider_data)):
        temp = rider_data[i].split(",")
        start = int(temp[0])
        end = int(temp[1])
        riders.append(Rider(start,end,0,random.randint(0,GROUPS-1)))
        riders[-1].value = get_ride_cost(riders[-1],A_coeff)+(random.random()-0.5)*A_coeff/4

    return riders

def get_initial_drivers(initial_drivers):
    locations = random.sample(driver_locations,initial_drivers)
    
    all_drivers = [Driver(0,TOTAL_EPOCHS,i) for i in locations]
    random.shuffle(all_drivers)

    return all_drivers

def get_groups(GROUPS):
    random.seed(0)
    k = [0 for i in range(GROUPS)]
    k_matrix = [[random.random()/10 for i in range(GROUPS)] for j in range(GROUPS)]
    for i in range(GROUPS):
        k_matrix[i][i] = random.random()/10+0.2
    return k,k_matrix


def update_drivers(drivers,all_drivers,epoch,data):
    for i in range(len(drivers)):
        if drivers[i].occupied and drivers[i].free_epoch<=epoch:
            drivers[i].set_occupied(False,-1)

    if epoch<=60:
        return drivers
    
    all_min_prices = set([i.min_price for i in drivers])
    non_active_drivers = [i for i in all_drivers if i.min_price not in all_min_prices]

    new_drivers = [i for i in drivers if i.occupied]

    previous_prices = []

    for e in range(epoch-60,epoch):
        end = e
        start = max(e-60,0)
        
        num_driven = data.serviced_riders_over_time[start:end]
        num_profit = data.profit_over_time[start:end]
        previous_prices.append(sum(num_profit)/sum(num_driven))

    previous_prices = sorted(previous_prices)
    median = previous_prices[len(previous_prices)//2]

    new_non_active = [i for i in non_active_drivers if i.min_price<=median]
    new_current = [i for i in drivers if not(i.occupied) and i.min_price<=median]

    return new_drivers + new_current+new_non_active


def get_current_state(drivers,epoch):
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
    return num_by_location+[idle_drivers,busy_drivers]+[epoch//60]

def update_state(new_state,rider,driver,k,k_matrix,valuation,price,epoch):
    g = rider.group
    new_state[regions[driver.location]]-=1
    new_state[regions[rider.end]]+=1
    new_state[num_regions+2] = (epoch+1)//60
    for group_num in range(len(k)):
        new_state[num_regions+3+group_num]+=k_matrix[group_num][g]*(valuation-price)
    new_state[num_regions]-=1
    new_state[num_regions+1]+=1

    return new_state


def solve_model(objective_values,m,n,riders,drivers,delta):
    model = Model()
    names = []
    objective = []

    for i in range(m):
        for j in range(n):
            variable = model.continuous_var(name='x{}_{}'.format(i, j))
            names.append(variable)
            objective.append(objective_values[(i,j)])
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
    
    solution = model.solve()

    matches = []
    
    for i in range(m):
        for j in range(n):
            if solution.get_value("x{}_{}".format(i,j)) == 1:
                matches.append((i,j))

    return matches

def new_k(matches,valuations,prices,k_matrix,riders):
    k_addition = [0 for i in range(len(k_matrix))]
    for (i,j) in matches:
        g = riders[i].group
        for group_num in range(len(k_matrix)):
            k_addition[group_num]+=k_matrix[group_num][g]*(valuations[i]-prices[i])
    return k_addition

def update_data(data,prices,costs,valuations,riders,drivers):
    data.num_occupied_drivers.append(len([i for i in drivers if i.occupied]))
    data.total_profit+=np.sum(list(prices.values()))-np.sum(list(costs.values()))
    data.revenue_over_time.append(np.sum(list(prices.values())))
    data.profit_over_time.append(np.sum(list(prices.values()))-np.sum(list(costs.values())))
    data.total_profit_over_time.append(data.total_profit)
    data.demand_over_time.append(len(riders))
    data.unsatisfied_demand_over_time.append(len(riders)-len(prices))
    data.serviced_riders_over_time.append(len(prices))
    data.num_drivers_over_time.append(len(drivers))

def move_drivers(riders,drivers,matches,epoch):
    for (i,j) in matches:
        rider = riders[i]
        dist = travel_times[rider.start][rider.end]
        time_to_get = travel_times[drivers[j].location][rider.start]
        drivers[j].set_occupied(True,time_to_get+dist+epoch)
        drivers[j].location = rider.end
