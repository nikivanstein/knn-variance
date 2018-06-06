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

folder = "npyNEW/"
benchmarkfunctions = [
					"rastrigin",
                    "ackley",
                    "schaffer"]

                    
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

n_init_sample = 100;
for n_init_sample in [50,100]:
	for dim in [2,5,10]:
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
			
			plt.figure(figsize=[5,6])
			markerset1 = ['o', 'v', '^', '<', '>']
			markerset2 = ['8', 's', 'p', '*', 'h']
			markerset3 = ['H', 'D', 'd', 'P', 'X']
			markerteller = -1
			for solver in ['CMA']: #, 'CMA-tree', 'BFGS-tree'
				markerteller += 1
				y_actual = optimals[f]
				#print "plotting ",f,dim
				

				ys = []
				times = []
				found = False
				for rank in range(40):

					
					if (os.path.isfile(folder+solver+f+"_GP_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy") ):
						#print "npy/"+solver+f+"_CK_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy"
						found = True
						y_hist,t = np.load(folder+solver+f+"_GP_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy")
						#t = np.load("npy/"+solver+f+"_GP_time_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy")
						times.append(t)
						#turn to RMSE
						ys.append(y_hist)
				if found:
					#if (solver=='BFGS'):
					timesGP.append(np.mean(times))
					timesGP_std.append(np.std(times))
					plt.errorbar(np.arange(1, len(ys[0])+1), np.mean(ys,axis=0), fmt='b-', label="Kriging var", marker='o', markersize=8,ecolor='#99c0ff',markevery=2)
					f1 = interp1d(np.arange(1, len(ys[0])+1),np.mean(ys,axis=0)-(np.std(ys,axis=0)) , kind='cubic')
					f2 = interp1d(np.arange(1, len(ys[0])+1),np.mean(ys,axis=0)+(np.std(ys,axis=0)) , kind='cubic')
					xnew = np.linspace(1,len(ys[0]),100)
					plt.fill_between(xnew, f1(xnew), f2(xnew), facecolor='#b7d2ff', alpha=0.4, interpolate=True)


				for rank in range(40):
					if (os.path.isfile(folder+solver+f+"_GPh_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy") ):
						#print "npy/"+solver+f+"_CK_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy"
						found = True
						y_hist,t = np.load(folder+solver+f+"_GPh_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy")
						#t = np.load("npy/"+solver+f+"_GP_time_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy")
						times.append(t)
						#turn to RMSE
						ys.append(y_hist)
				if found:
					timesGPv.append(np.mean(times))
					timesGPv_std.append(np.std(times))
					plt.errorbar(np.arange(1, len(ys[0])+1), np.mean(ys,axis=0), fmt='g-', label="Kriging k-NN var", marker='D', markersize=8,ecolor='#99ffc0',markevery=2)
					f1 = interp1d(np.arange(1, len(ys[0])+1),np.mean(ys,axis=0)-(np.std(ys,axis=0)) , kind='cubic')
					f2 = interp1d(np.arange(1, len(ys[0])+1),np.mean(ys,axis=0)+(np.std(ys,axis=0)) , kind='cubic')
					xnew = np.linspace(1,len(ys[0]),100)
					plt.fill_between(xnew, f1(xnew), f2(xnew), facecolor='#b7ffd2', alpha=0.4, interpolate=True)
					
				'''
				ys = []
				times = []
				for rank in range(20):
					if (os.path.isfile("npy/"+solver+f+"_GMM_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy") ):
						y_hist = np.load("npy/"+solver+f+"_GMM_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy")
						t = np.load("npy/"+solver+f+"_GMM_time_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy")
						times.append(t)
						#turn to RMSE
						ys.append(y_hist)
				timesGMM.append(np.mean(times))
				timesGMM_std.append(np.std(times))
				plt.errorbar(np.arange(1, len(ys[0])+1), np.mean(ys,axis=0), fmt='y-', label="GMMCK", marker="^", markersize=14,ecolor='#ffee99')
				f1 = interp1d(np.arange(1, len(ys[0])+1),np.mean(ys,axis=0)-np.std(ys,axis=0) , kind='cubic')
				f2 = interp1d(np.arange(1, len(ys[0])+1),np.mean(ys,axis=0)+np.std(ys,axis=0) , kind='cubic')
				xnew = np.linspace(1,len(ys[0]),100)
				plt.fill_between(xnew, f1(xnew), f2(xnew), facecolor='#fffdb7', alpha=0.4, interpolate=True)
				'''

				ys = []
				times = []
				found = False
				for rank in range(40):
					if (os.path.isfile(folder+solver+f+"_RF_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy") ):
						found = True
						y_hist,t = np.load(folder+solver+f+"_RF_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy")
						times.append(t)
						#turn to RMSE
						ys.append(y_hist)
				if(found):
					timesRF.append(np.mean(times))
					timesRF_std.append(np.std(times))
					plt.errorbar(np.arange(1, len(ys[1])+1), np.mean(ys,axis=0), fmt='r:', label="RF k-NN var", marker='*', markersize=8,ecolor='#ff999a',markevery=2)
					f1 = interp1d(np.arange(1, len(ys[1])+1),np.mean(ys,axis=0)-(np.std(ys,axis=0)) , kind='cubic')
					f2 = interp1d(np.arange(1, len(ys[1])+1),np.mean(ys,axis=0)+(np.std(ys,axis=0)) , kind='cubic')
					xnew = np.linspace(1,len(ys[0]),100)
					plt.fill_between(xnew, f1(xnew), f2(xnew), facecolor='#ffd1b7', alpha=0.4, interpolate=True)
					

				ys = []
				times = []
				found = False
				for rank in range(40):
					if (os.path.isfile(folder+solver+f+"_ANN_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy") ):
						found = True
						y_hist,t = np.load(folder+solver+f+"_ANN_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy")
						times.append(t)
						#turn to RMSE
						ys.append(y_hist)
				if(found):
					timesANN.append(np.mean(times))
					timesANN_std.append(np.std(times))
					plt.errorbar(np.arange(1, len(ys[1])+1), np.mean(ys,axis=0), fmt='y:', label="ANN k-NN var", marker='<', markersize=8,ecolor='#ffff88',markevery=2)
					f1 = interp1d(np.arange(1, len(ys[1])+1),np.mean(ys,axis=0)-(np.std(ys,axis=0)) , kind='cubic')
					f2 = interp1d(np.arange(1, len(ys[1])+1),np.mean(ys,axis=0)+(np.std(ys,axis=0)) , kind='cubic')
					xnew = np.linspace(1,len(ys[0]),100)
					plt.fill_between(xnew, f1(xnew), f2(xnew), facecolor='#ffff88', alpha=0.4, interpolate=True)
					



				ys = []
				times = []
				found = False
				for rank in range(40):
					if (os.path.isfile(folder+solver+f+"_K_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy") ):
						found = True
						y_hist,t = np.load(folder+solver+f+"_K_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy")
						#t = np.load("npy/"+solver+f+"_K_time_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy")
						times.append(t)
						ys.append(y_hist)
				if(found):
					plt.errorbar(np.arange(1, len(ys[0])+1), np.mean(ys,axis=0), fmt='#222222', label="OK--"+solver, marker=markerset3[markerteller], markersize=8, ecolor="#AAAAAA",markevery=2)
					f1 = interp1d(np.arange(1, len(ys[0])+1),np.mean(ys,axis=0)-(np.std(ys,axis=0)) , kind='cubic')
					f2 = interp1d(np.arange(1, len(ys[0])+1),np.mean(ys,axis=0)+(np.std(ys,axis=0)) , kind='cubic')
					xnew = np.linspace(1,len(ys[0]),100)
					plt.fill_between(xnew, f1(xnew), f2(xnew), facecolor='#CCCCCC', alpha=0.4, interpolate=True)

					if (solver=='BFGS'):
						timesk.append(np.mean(times))
						timesk_std.append(np.std(times))
					plt.xlim([0,len(ys[0])+1])
			#plt.grid(True)
			plt.legend()
			plt.savefig("imgNEW/"+f+"_"+`dim`+"_"+`n_init_sample`+".png", bbox_inches='tight')


		if True:

			n_groups = len(benchmarkfunctions)

			fig, ax = plt.subplots(figsize=[12,8])
			index = np.arange(n_groups)
			bar_width = 0.20
			opacity = 0.4
			error_config = {'ecolor': '0.3'}
			print "Kriging &",dim,
			for i in range(len(timesGP)):
				print "&",timesGP[i],"+/-",timesGP_std[i],
			print ""
			#print(index,timesck,timesck_std)
			rects1 = plt.bar(index, timesGP, bar_width,
			                 alpha=opacity,
			                 color='b',
			                 yerr=timesGP_std,
			                 error_kw=error_config,
			                 label='Kriging var.')

			print "Kriging k-NN &",dim,
			for i in range(len(timesGPv)):
				print "&",timesGPv[i],"+/-",timesGPv_std[i],
			print ""
			rects2 = plt.bar(index + bar_width, timesGPv, bar_width,
			                 alpha=opacity,
			                 color='#222222',
			                 yerr=timesGPv_std,
			                 error_kw=error_config,
			                 label='Kriging k-NN')

			'''
			rects3 = plt.bar(index + bar_width*2, timesGMM, bar_width,
			                 alpha=opacity,
			                 color='y',
			                 yerr=timesGMM_std,
			                 error_kw=error_config,
			                 label='GMMCK')
			 '''
			if False:
				print "RF k-NN &",dim,
				for i in range(len(timesRF)):
					print "&",timesRF[i],"+/-",timesRF_std[i],
				print ""
				rects3 = plt.bar(index + bar_width*2, timesRF, bar_width,
				                 alpha=opacity,
				                 color='g',
				                 yerr=timesRF_std,
				                 error_kw=error_config,
				                 label='RF k-NN')

			print "ANN k-NN &",dim,
			for i in range(len(timesRF)):
				print "&",timesANN[i],"+/-",timesANN_std[i],
			print ""
			rects3 = plt.bar(index + bar_width*2, timesANN, bar_width,
			                 alpha=opacity,
			                 color='y',
			                 yerr=timesANN_std,
			                 error_kw=error_config,
			                 label='ANN k-NN')

			plt.xlabel('Benchmark function')
			plt.ylabel('Average CPU time')
			plt.title('Average CPU time per Benchmark')
			plt.xticks(index + bar_width / 2, benchmarkfunctions)
			plt.legend()
			plt.grid(True)
			#plt.tight_layout()
			plt.savefig("imgNEW/time"+"_"+`dim`+"_"+`n_init_sample`+".png",bbox_inches='tight')
			#except Exception:
			#	continue
	


#plt.show()