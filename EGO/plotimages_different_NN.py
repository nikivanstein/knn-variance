import pdb
import sys
import numpy as np
from numpy.random import rand
from numpy import zeros, ones, array, sqrt, size, nonzero, min, max, log, sum, inf
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os.path
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.interpolate import interp1d

plt.ion()
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['font.size'] = 15
rcParams['legend.numpoints'] = 1 
rcParams['xtick.labelsize'] = 13
rcParams['ytick.labelsize'] = 13
rcParams['xtick.major.size'] = 7
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.size'] = 7
rcParams['ytick.major.width'] = 1

plt.ioff()

folder = "npyANN/"
benchmarkfunctions = ["ackley"]

                    
					#"schwefel",
					#"schaffer",


optimals = {"rastrigin":0.0,
					"schwefel":0.0,
					"schaffer":0.0,
                    "ackley":0.0,
                    "griewank":0.0,
                   "bohachevsky":0.0}



markers = ['*','D']

methods = ["ANN","CK","K","GMM",'k-means']

n_init_sample = 1000;
marker_sNN = 0
solver = "CMA"
plt.figure(figsize=[12,6])
for sNN in [2,5,10,15,20,25,50]:
	for n_init_sample in [100]:
		for dim in [10]:
			timesGP = []
			timesGPv = []
			timesRF = []
			timesKCK = []

			timesANN = []
			timesANN_std = []
			timesGP_std = []
			timesGPv_std = []
			timesRF_std = []
			timesKCK_std = []
			for f in benchmarkfunctions:
				
				
				markerset1 = ['o', 'v', '^', '<', '>','8', 's', 'p', '*', 'h','H', 'D', 'd', 'P', 'X']
				markerteller = -1
			
				ys = []
				times = []
				found = False
				for rank in range(40):
					if (os.path.isfile(folder+solver+f+"_ANN_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+"_"+`sNN`+".npy") ):
						found = True
						y_hist,t = np.load(folder+solver+f+"_ANN_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+"_"+`sNN`+".npy")
						times.append(t)
						ys.append(y_hist)
				if(found):
					timesANN.append(np.mean(times))
					timesANN_std.append(np.std(times))
					plt.errorbar(np.arange(1, len(ys[1])+1), np.mean(ys,axis=0), label="k= "+`sNN`, marker=markerset1[marker_sNN], markersize=8,markevery=5)
					f1 = interp1d(np.arange(1, len(ys[1])+1),np.mean(ys,axis=0)-(np.std(ys,axis=0)) , kind='cubic')
					f2 = interp1d(np.arange(1, len(ys[1])+1),np.mean(ys,axis=0)+(np.std(ys,axis=0)) , kind='cubic')
					xnew = np.linspace(1,len(ys[0]),100)
					#plt.fill_between(xnew, f1(xnew), f2(xnew), facecolor='#FEFEFE', alpha=0.4, interpolate=True)
					marker_sNN += 1



#plt.grid(True)
plt.legend()
plt.savefig("img/ANN_ackley_10_100_differentNN.png", bbox_inches='tight')

