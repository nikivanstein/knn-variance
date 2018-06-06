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


benchmarkfunctions = ["rastrigin"]

optimals = {"rastrigin":0.0,
					"schwefel":0.0,
					"schaffer":0.0,
                    "ackley":0.0,
                    "griewank":0.0,
                   "bohachevsky":0.0}



markers = ['*','D']

n_init_sample = 1000;

for n_init_sample in [500,1000,5000]:
	for dim in [2,5,10]:
		markers = ["*","+","d"]
		cluster_i = 0
		timesall = []
		for clusters in [5,10,15]:
			timesck = []
			print dim,clusters

			timesck_std = []
			timesk_std = []
			for f in benchmarkfunctions:
				y_actual = optimals[f]
				print "plotting ",f,dim
				ys = []
				times = []
				plt.figure(figsize=[6,8])
				for rank in range(20):
					if (os.path.isfile("npy/"+f+"_CK_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+"_"+`clusters`+".npy") ):
						y_hist = np.load("npy/"+f+"_CK_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+"_"+`clusters`+".npy")
						t = np.load("npy/"+f+"_CK_time_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+"_"+`clusters`+".npy")
						times.append(t)
						#turn to RMSE
						rmse = []
						for y in y_hist:
							rmse.append( y )
						ys.append(rmse)
				timesck.append(np.mean(times))
				timesck_std.append(np.std(times))

				timesall.append(timesck)
				timesall.append(timesck_std)

				#sns.tsplot(data=ys, label="MTCK")
				plt.errorbar(np.arange(1, len(ys[0])+1), np.mean(ys,axis=0),yerr=np.std(ys,axis=0), fmt='b-', label="MTCK-"+`clusters`, marker=markers[cluster_i], markersize=14,ecolor='#99c0ff')
				cluster_i += 1

				f1 = interp1d(np.arange(1, len(ys[0])+1),np.mean(ys,axis=0)-np.std(ys,axis=0) , kind='cubic')
				f2 = interp1d(np.arange(1, len(ys[0])+1),np.mean(ys,axis=0)+np.std(ys,axis=0) , kind='cubic')
				xnew = np.linspace(1,len(ys[0]),100)
				
				plt.fill_between(xnew, f1(xnew), f2(xnew), facecolor='#b7d2ff', alpha=0.7, interpolate=True)
				
				plt.xlim([0,len(ys[0])+1])
				#plt.grid(True)
				plt.legend()
				plt.savefig("img/clusters_"+f+"_"+`dim`+"_"+`n_init_sample`+".png", bbox_inches='tight')


		n_groups = 3
		#print timesck

		fig, ax = plt.subplots(figsize=[12,8])
		index = np.arange(n_groups)
		bar_width = 0.30
		opacity = 0.4
		error_config = {'ecolor': '0.3'}
		rects1 = plt.bar(index+ bar_width*0, timesall[0], bar_width,
		                 alpha=opacity,
		                 color='b',
		                 yerr=timesck_std,
		                 error_kw=error_config,
		                 label='MTCK-2')

		rects2 = plt.bar(index+ bar_width*1, timesall[1], bar_width,
		                 alpha=opacity,
		                 color='b',
		                 yerr=timesck_std,
		                 error_kw=error_config,
		                 label='MTCK-5')
		rects3 = plt.bar(index+ bar_width*2, timesall[2], bar_width,
		                 alpha=opacity,
		                 color='b',
		                 yerr=timesck_std,
		                 error_kw=error_config,
		                 label='MTCK-10')

		plt.xlabel('Benchmark function')
		plt.ylabel('Average CPU time')
		plt.title('Average CPU time per Benchmark')
		plt.xticks(index + bar_width / 2, benchmarkfunctions)
		plt.legend()
		plt.grid(True)
		#plt.tight_layout()
		plt.savefig("img/clusters_time"+"_"+`dim`+"_"+`n_init_sample`+".png",bbox_inches='tight')
		


#plt.show()