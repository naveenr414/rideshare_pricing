import random
import numpy as np
from math import e
from sklearn.cluster import KMeans


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
    return  2/(1+e**(-x))-1
driver_locations = open("data/taxi_3000_final.txt").read().strip().split("\n")
driver_locations = [int(i) for i in driver_locations]

ride_data = open("data/test_flow_5000_1.txt")
TOTAL_EPOCHS = int(ride_data.readline())
ride_data.readline()

time_per_match = 60
travel_times = np.load('data/zone_traveltime.npy')
travel_times = np.round(travel_times/time_per_match)

def read_riders(A_coeff,M_coeff,GROUPS):
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
        valuation = A_coeff*travel_times[start][end]+(random.random()-0.5)*A_coeff/4
        riders.append(Rider(start,end,valuation,random.randint(0,GROUPS-1)))

    return riders

def get_initial_drivers(initial_drivers):
    return [Driver(0,TOTAL_EPOCHS,i) for i in driver_locations[:initial_drivers]]

def get_k_regions(num_regions):
    a = open("data/zone_latlong.csv").read().strip().split("\n")    
    a = [[i.split(",")[1],i.split(",")[2]] for i in a]
    a = [[float(i[0]),float(i[1])] for i in a]
    a = np.array(a)

    kmeans = KMeans(n_clusters=num_regions).fit(a)
    return kmeans.labels_
