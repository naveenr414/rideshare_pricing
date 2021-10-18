import random
import numpy as np
from math import e
from sklearn.cluster import KMeans
from docplex.mp.model import Model  # type: ignore
from docplex.mp.linear import Var  # type: ignore
import matplotlib.pyplot as plt
import pickle

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
        self.driver_extra_pay_over_time = []

    def __dict__(self):
        return {'total_profit': self.total_profit,
                'profit_over_time': self.profit_over_time,
                'demand_over_time': self.demand_over_time,
                'unsatisfied_demand_over_time': self.unsatisfied_demand_over_time,
                'num_occupied_drivers': self.num_occupied_drivers,
                'serviced_riders_over_time': self.serviced_riders_over_time,
                'revenue_over_time': self.revenue_over_time,
                'total_profit_over_time': self.total_profit_over_time,
                'num_drivers_over_time': self.num_drivers_over_time,
                'driver_extra_pay_over_time': self.driver_extra_pay_over_time}

class Driver:
    def __init__(self,time_in,time_out,initial_location,driver_opportunity_cost_avg):
        self.time_in = time_in
        self.time_out = time_out
        self.initial_location = initial_location
        self.occupied = False
        self.free_epoch = -1
        self.location = initial_location
        self.opportunity_cost_per_hour = driver_opportunity_cost_avg*np.random.normal(1,0.3)

    def set_occupied(self,occupied,free_epoch):
        self.occupied = occupied
        self.free_epoch = free_epoch

class Rider:
    def __init__(self,start,end,value,group):
        self.start = start
        self.end = end
        self.time_taken = travel_times[start][end]
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

future_demand = pickle.load(open("time_model_1_hour.p","rb"))

driver_locations = open("data/taxi_3000_final.txt").read().strip().split("\n")
driver_locations = [int(i) for i in driver_locations]

all_ride_data = open("data/test_flow_5000_1.txt").read().strip().split("\n")[1:]
all_ride_data = [i for i in all_ride_data if 'Flows' not in i]

day_num = 1
ride_data = open("data/test_flow_5000_{}.txt".format(day_num))
TOTAL_EPOCHS = int(ride_data.readline())
ride_data.readline()

time_per_match = 60
travel_times = np.load('data/zone_traveltime.npy')
travel_times = np.round(travel_times/time_per_match)

num_regions = 5
regions = get_k_regions(num_regions)

def get_ride_cost(rider,rider_avg_price):
    time_taken = travel_times[rider.start][rider.end]
    avg_cost = (time_taken/60)*rider_avg_price

    return np.random.normal(1,0.5)*avg_cost

def get_valuation(rider,M_coeff,k):
    return rider.value+M_coeff*sigmoid(k[rider.group])

def process_ride_data(rider_data,rider_avg_price,GROUPS):
    riders = []
    for i in range(len(rider_data)):
        temp = rider_data[i].split(",")
        start = int(temp[0])
        end = int(temp[1])
        new_rider = Rider(start,end,0,random.randint(0,GROUPS-1))
        new_rider.value = get_ride_cost(new_rider,rider_avg_price)
        riders.append(new_rider)

    return riders

def random_rides(num_rides,rider_avg_price,GROUPS):
    selected_rides = random.sample(all_ride_data,num_rides)
    return process_ride_data(selected_rides,rider_avg_price,GROUPS)

def read_riders(rider_avg_price,GROUPS):
    rider_data = []
    line = ride_data.readline()
    while "Flows" not in line and line!='':
        rider_data.append(line)
        line = ride_data.readline()

    return process_ride_data(rider_data,rider_avg_price,GROUPS)

def get_initial_drivers(initial_drivers,driver_opportunity_cost_avg):
    locations = random.sample(driver_locations,initial_drivers)
    
    all_drivers = [Driver(0,TOTAL_EPOCHS,i,driver_opportunity_cost_avg) for i in locations]
    random.shuffle(all_drivers)

    return all_drivers

def get_groups(GROUPS):
    random.seed(0)
    k = [0 for i in range(GROUPS)]
    k_matrix = [[random.random()/10 for i in range(GROUPS)] for j in range(GROUPS)]
    for i in range(GROUPS):
        k_matrix[i][i] = random.random()/10+0.2
    return k,k_matrix

def reset_day(new_day_num):
    global day_num
    global ride_data
    
    day_num = 1
    ride_data = open("data/test_flow_5000_{}.txt".format(day_num))
    ride_data.readline()

def update_drivers(drivers,all_drivers,epoch,data,driver_comission):
    # Look at the amount earned per hour by drivers
    # Is this more or less than the opportunity cost 
    
    for i in range(len(drivers)):
        if drivers[i].occupied and drivers[i].free_epoch<=epoch:
            drivers[i].set_occupied(False,-1)
    new_drivers = [i for i in drivers if i.occupied]

    if epoch<=60:
        return drivers
    
    all_min_prices = set([i.opportunity_cost_per_hour for i in drivers])
    non_active_drivers = [i for i in all_drivers if i.opportunity_cost_per_hour not in all_min_prices]

    previous_revenues = []

    for end in range(epoch-60,epoch):
        start = max(end-60,0)
        avg_num_drivers = np.mean(data.num_drivers_over_time[start:end])
        total_driver_revenue = np.sum(data.revenue_over_time[start:end])*driver_comission+np.sum(data.driver_extra_pay_over_time[start:end])
        driver_average_revenue = total_driver_revenue/avg_num_drivers
        driver_average_revenue/=((end-start)/60)
        previous_revenues.append(driver_average_revenue)
        
    previous_revenues = sorted(previous_revenues)
    median_revenue = previous_revenues[len(previous_revenues)//2]

    new_non_active = [i for i in non_active_drivers if (i.opportunity_cost_per_hour<=median_revenue or random.random()<.005)]

    for i in new_non_active:
        i.time_in = epoch

    new_current = []
    for i in non_active_drivers:
        stay = i.opportunity_cost_per_hour<=median_revenue or abs(i.time_in-epoch)<=60
        rand_leave = (random.random()>.0025)
        stay = stay and rand_leave
        if stay:
            new_current.append(i)
        else:
            i.time_out = epoch
    
    drivers = new_drivers + new_current+new_non_active
    
    return drivers

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

def update_data(data,prices,costs,driver_extra_pay,valuations,riders,drivers):
    data.num_occupied_drivers.append(len([i for i in drivers if i.occupied]))
    data.total_profit+=np.sum(list(prices.values()))-np.sum(list(costs.values()))-np.sum(list(driver_extra_pay.values()))
    data.driver_extra_pay_over_time.append(np.sum(list(driver_extra_pay.values())))
    data.revenue_over_time.append(np.sum(list(prices.values())))
    data.profit_over_time.append(np.sum(list(prices.values()))-np.sum(list(costs.values()))--np.sum(list(driver_extra_pay.values())))
    data.total_profit_over_time.append(data.total_profit)
    data.demand_over_time.append(len(riders))
    data.unsatisfied_demand_over_time.append(len(riders)-len(prices))
    data.serviced_riders_over_time.append(len(prices))
    data.num_drivers_over_time.append(len(drivers))

def predict_future_demand(data,epoch):
    l = []
    for mins_ago in [30,35,40,45,50,55,60]:
        l.append(data.demand_over_time[-mins_ago])
    l.append(epoch)
    return future_demand.predict([l])[0]

def move_drivers(riders,drivers,matches,epoch):
    for (i,j) in matches:
        rider = riders[i]
        dist = travel_times[rider.start][rider.end]
        time_to_get = travel_times[drivers[j].location][rider.start]
        drivers[j].set_occupied(True,time_to_get+dist+epoch)
        drivers[j].location = rider.end

def rolling_average(l,num=60):
    all_nums = []
    for i in range(num,len(l)):
            all_nums.append(np.mean(l[i-num:i]))
    return all_nums
