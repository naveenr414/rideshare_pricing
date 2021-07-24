import glob
import pickle

all_results = glob.glob("results/*")

all_dicts = []
for i in all_results:
    f = pickle.load(open(i,'rb'))
    all_dicts.append(f)
