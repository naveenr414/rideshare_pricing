import glob
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor

def get_all_results(location):
    all_results = glob.glob(location)

    all_dicts = []
    for i in all_results:
        f = pickle.load(open(i,'rb'))
        if 'externality_multiplier' in f.keys():
            all_dicts.append(f)

    return all_dicts

def get_model_number(d):
    return int(d['model_name'].split(".")[0].split("_")[-1])

def get_num_riders_over_time(day_num):
    ride_data = open("data/test_flow_5000_{}.txt".format(day_num))
    ride_data.readline()
    ride_data.readline()
    riders_by_epoch = []
    for i in range(1440):
            line = ride_data.readline()
            riders = 0
            while "Flows" not in line and line!='':
                    line = ride_data.readline()
                    riders+=1
            riders_by_epoch.append(riders)
    return riders_by_epoch

def get_X(rider_data):
    X = []
    y = []
    for i in range(len(rider_data)):
        if i<60:
            continue

        l = []
        for mins_ago in [30,35,40,45,50,55,60]:
            l.append(rider_data[i-mins_ago])
        l.append(i)
        X.append(l)
        y.append(rider_data[i])
    return np.array(X), np.array(y)

def get_train_data():
    X_list = []
    y_list = []

    for i in range(1,10):
        X,y = get_X(get_num_riders_over_time(i))
        X_list.append(X)
        y_list.append(y)
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    return X,y

def get_test_data():
    return get_X(get_num_riders_over_time(10))

X,y = get_train_data()
X_test, y_test = get_test_data()
regr = MLPRegressor(max_iter=500).fit(X,y)
y_predict = regr.predict(X_test)

plt.plot(range(60,len(y_predict)+60),y_predict,label="Predicted")
plt.plot(range(60,len(y_predict)+60),y_test,label="Actual")
plt.legend()
plt.show()
pickle.dump(regr,open("time_model_1_hour.p","wb"))

