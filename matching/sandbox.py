import glob
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np

def get_all_results(location):
    all_results = glob.glob(location)

    all_dicts = []
    for i in all_results:
        f = pickle.load(open(i,'rb'))
        if 'externality_multiplier' in f.keys():
            all_dicts.append(f)

    return all_dicts

all_dicts = get_all_results("results/noisiness_runs/*.p")

profit_distribution = [i['total_profit'] for i in all_dicts]
print(np.mean(profit_distribution),np.std(profit_distribution))
