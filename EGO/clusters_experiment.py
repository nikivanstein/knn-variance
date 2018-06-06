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

from OWCK import GaussianProcess_extra as GaussianProcess
from OWCK import OWCK
from ego import ego

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
benchmarkfunctions = {
                "rastrigin":benchmarks.rastrigin
}
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
n_step = 15
dim = 2
#clusters = 5
for clusters in [5,10,15]:
    for n_init_sample in [500,1000,5000]:
        #print("samples:",n_init_sample)
        for dim in [2]:
            for key, bfunc in benchmarkfunctions.iteritems():
                print ("function:",key,"dim:",dim, "samples:",n_init_sample, "clusters",clusters)

                thetaL = 1e-3 * np.ones(dim) * (functionbounds[key][1] - functionbounds[key][0]) 
                thetaU = 10 * np.ones(dim) * (functionbounds[key][1] - functionbounds[key][0]) 
                theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL
                

                start_time = time.time()
                y_hist_list = []
                test_fun = lambda x: bfunc(x)[0]
                func_name = key
                success = False
                #OWCK with n/100 clusters
                import pdb
                #pdb.set_trace()
                model = OWCK(regr='constant', corr='matern', cluster_method="tree", theta0=theta0, thetaL=thetaL, thetaU=thetaU,
                      n_cluster=clusters, nugget=None, verbose=False, random_start=10, 
                      nugget_estim=True,
                      is_parallel=False)
                
                optimizer = ego(dim, test_fun, model, n_step,
                    lb=np.ones(dim)*functionbounds[key][0],
                    ub=np.ones(dim)*functionbounds[key][1], 
                    doe_size=n_init_sample,
                    solver="BFGS",
                    verbose=False,
                    random_seed=rank)


                xopt, fopt, _, __, y_hist = optimizer.optimize()
                success = True
                #print y_hist
                 
                np.save("/data/steinbvan/npy/"+key+"_CK_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+"_"+`clusters`+".npy",y_hist)
                timeck = time.time()-start_time
                np.save("/data/steinbvan/npy/"+key+"_CK_time_"+`rank`+"_"+`n_init_sample`+"_"+`dim`+"_"+`clusters`+".npy",timeck)
                #print "time", timeck
                
                 

                 
        print(key, "Finished one run in:",time.time()-start_time )