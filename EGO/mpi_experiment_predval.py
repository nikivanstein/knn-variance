# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 12:31:07 2014

@author: wangronin, steinbvan


"""
import pdb
import sys
import numpy as np
from copy import deepcopy
from pyDOE import lhs

from scipy.stats import norm
from numpy.random import rand
from numpy import ones, array, sqrt, nonzero, min, max, sum, inf

# optimizers
from cma_es_class import cma_es
from scipy.optimize import fmin_l_bfgs_b

import random
import warnings

from mpi4py import MPI
import time
import fitness
from deap import benchmarks

from sklearn.gaussian_process import GaussianProcessRegressor
from owck import GaussianProcess_extra as GaussianProcess
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from ego import ego as ego_old
from ego_pv import ego



folder = "/data/steinbvan/predvar/"

optimums = {}
optimums["diffpow"]=0
optimums["ackley"]=0
optimums["himmelblau"]=0
optimums["rastrigin"]=0
optimums["schwefel"]=0

size = MPI.COMM_WORLD.Get_size() 
rank = MPI.COMM_WORLD.Get_rank() 
name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

 
M = 4 #number of runs #not used, because we use rank in MPI mode
hist_best = []
#hartman y_opt = -log(-(-3.32237))
y_opt = 0 #0.397887
#"schwefel":benchmarks.schwefel,
 #               "ackley":benchmarks.ackley,
 #               "schaffer":benchmarks.schaffer
benchmarkfunctions = {"ackley":benchmarks.ackley,
                "rastrigin":benchmarks.rastrigin,
                "schaffer":benchmarks.schaffer} #"schaffer":benchmarks.schaffer
                #"ackley":benchmarks.ackley,
                #"rastrigin":benchmarks.rastrigin,
                #"schaffer":benchmarks.schaffer}
functionbounds = {"schwefel":[-500,500],
                "ackley":[-15,30],
                "rastrigin":[-5.12,5.12],
                "schaffer":[-100,100]}

func_name = "schwefel" #see on top of file
#removed #lambda x:  hardman
#test_fun = lambda x: benchmarks.ackley(x)[0]
#test_fun = fitness.branin


strategy = "niching" # CL or niching
n_init_sample = 1000
n_step = 25
dim = 2
clusters = 5
sample_sizes = [50,100]
for sNN in [20]: #2,5,10,15,20,25,50
    for n_init_sample in sample_sizes:
        for solver in ["CMA"]: #"CMA",
            print("samples:",n_init_sample)
            for dim in [2,5,10]:
                n_step = dim * 10
                for key, bfunc in benchmarkfunctions.iteritems():
                    func_start_time = time.time()
                    print ("function:",key,"dim:",dim,"solver:",solver,"samples:",n_init_sample)

                    thetaL = 1e-3 * np.ones(dim) * (functionbounds[key][1] - functionbounds[key][0]) 
                    thetaU = 10 * np.ones(dim) * (functionbounds[key][1] - functionbounds[key][0]) 
                    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL
                    test_fun = lambda x: bfunc(x)[0]
                    func_name = key
                    y_hist_list = []
                    if (True):
                        
                        #First test a Guassian process as normal
                        start_time = time.time()
                        
                        
                        model = GaussianProcess(corr='matern',thetaL=thetaL,thetaU=thetaU,theta0=theta0, nugget=None, normalize=False, nugget_estim=False)     
                        optimizer = ego_old(dim, test_fun, model, n_step,
                                lb=np.ones(dim)*functionbounds[key][0],
                                ub=np.ones(dim)*functionbounds[key][1], 
                                doe_size=n_init_sample,
                                solver=solver,
                                verbose=False,
                                random_seed=rank)
                        #print "start optimizing"
                        xopt, fopt, _, __, y_hist = optimizer.optimize()
                        success = True
                            
                         
                        timeck = time.time()-start_time
                        np.save(folder+solver+key+"_GP_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy",[y_hist,timeck])
                        
                    if (True):
                        start_time = time.time()
                        #kernel = C(1.0, (1e-5, 1e5)) * RBF(10, (1e-5, 1e5))
                        #model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
                        model = GaussianProcess(corr='matern',thetaL=thetaL,thetaU=thetaU,theta0=theta0, nugget=None, normalize=False, nugget_estim=False)
                        optimizer = ego(dim, test_fun, model, n_step,
                                lb=np.ones(dim)*functionbounds[key][0],
                                ub=np.ones(dim)*functionbounds[key][1], 
                                doe_size=n_init_sample,
                                solver=solver,
                                verbose=False,
                                random_seed=rank)
                        #print "start optimizing"
                        xopt, fopt, _, __, y_hist = optimizer.optimize()
                        success = True
                            
                         
                        timeck = time.time()-start_time
                        np.save(folder+solver+key+"_GPh_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy",[y_hist,timeck])
                        #np.save(folder+solver+key+"_GP_time_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy",timeck)
                     
                    if False:
                        start_time = time.time()
                        #try:
                        print "RF", key, rank

                        model = RandomForestRegressor(n_estimators=100)
                        optimizer = ego(dim, test_fun, model, n_step,
                                lb=np.ones(dim)*functionbounds[key][0],
                                ub=np.ones(dim)*functionbounds[key][1], 
                                doe_size=n_init_sample,
                                solver=solver,
                                verbose=False,
                                random_seed=rank)
                        xopt, fopt, _, __, y_hist = optimizer.optimize()
                        #except:
                        #    print "Caught it!, trying again"
                        print "done RF", key, rank
                        
                        timeck = time.time()-start_time
                        np.save(folder+solver+key+"_RF_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy",[y_hist,timeck])
                        #np.save(folder+solver+key+"_RF_time_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy",timeck)
                        #
                    if False:
                        start_time = time.time()
                        #TODO: build a multi layer perceptron
                        print "ANN", key, rank

                        model = MLPRegressor(hidden_layer_sizes=(100,50 ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                        optimizer = ego(dim, test_fun, model, n_step,
                                lb=np.ones(dim)*functionbounds[key][0],
                                ub=np.ones(dim)*functionbounds[key][1], 
                                doe_size=n_init_sample,
                                solver=solver,
                                verbose=False,
                                sNN = sNN,
                                random_seed=rank)
                        xopt, fopt, _, __, y_hist = optimizer.optimize()
                        #except:
                        #    print "Caught it!, trying again"
                        print "done ANN", key, rank
                        
                        timeck = time.time()-start_time
                        np.save(folder+solver+key+"_ANN_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy",[y_hist,timeck]) #+"_"+`sNN`
                        #np.save(folder+solver+key+"_RF_time_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+".npy",timeck)


                    #todo RF bootstrapping
                    
                     
                    print(key, "Finished one run in:",time.time()-func_start_time )