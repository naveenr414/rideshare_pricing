import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

def num_drivers_by_epoch(file_name):
	f = open(file_name)
	line = f.readline()
	num_by_epoch = []
	while line.strip()!='':
		line = f.readline()
		num = 0
		while "Flows" not in line and line.strip()!='':
			num+=1
			line = f.readline()
		num_by_epoch.append(num)
	return num_by_epoch[1:]

all_results = glob.glob("results/*.p")

all_dicts = []
for i in all_results:
    f = pickle.load(open(i,'rb'))
    if 'externality_multiplier' in f.keys():
        all_dicts.append(f)
        print(f.keys())


pairs = {}
for i in all_dicts:
    pairs[(i['frictional_multiplier'],i['externality_multiplier'])] = i

frictional_values = [0.9,1,1.1,1.5,2]
frictional_multiplier = [(i,2) for i in frictional_values]

externality_values = [1,2,4,8,16]
externality_multiplier = [(1,j) for j in externality_values]

num_drivers_frictional = []
num_drivers_exteranlity = []

for p in frictional_multiplier:
    num_drivers_frictional.append(np.mean(pairs[p]['num_drivers_over_time']))

for p in externality_multiplier:
    num_drivers_exteranlity.append(np.mean(pairs[p]['num_drivers_over_time']))

plt.plot(num_drivers_frictional)
plt.xticks(range(len(frictional_values)),frictional_values,fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Frictional Multiplier (F)",fontsize=16)
plt.ylabel("Average number of drivers",fontsize=16)
plt.title("Frictional Multiplier vs. num drivers",fontsize=20)
plt.tight_layout()
plt.savefig("images/f_drivers.png")
plt.show()

plt.plot(num_drivers_exteranlity)
plt.xticks(range(len(externality_values)),externality_values,fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Externality Multiplier (M)",fontsize=16)
plt.ylabel("Average number of drivers",fontsize=16)
plt.title("Externality Multiplier vs. num drivers",fontsize=20)
plt.tight_layout()
plt.savefig("images/m_drivers.png")
plt.show()
