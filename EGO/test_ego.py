# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 17:21:45 2017

@author: wangronin
"""

import os

from mpi4py import MPI
import pandas as pd

import numpy as np
from numpy import ones

from owck import OWCK

from deap import benchmarks
from ego import ego


def create_test(dim, n_init_sample, fitness_func, n_step, method, n_cluster, seed):
    
    x_lb = np.array([functionbounds[func_name][0]] * dim)
    x_ub = np.array([functionbounds[func_name][1]] * dim)
    
    length_lb = 1e-10
    length_ub = 1e2
    thetaL = length_ub ** -2. / (x_ub - x_lb) ** 2. * ones(dim)
    thetaU = length_lb ** -2. / (x_ub - x_lb) ** 2. * ones(dim)
        
    # initial search point for hyper-parameters
    theta0 = np.random.rand(1, dim) * (thetaU - thetaL) + thetaL

    model = OWCK(corr='matern', n_cluster=n_cluster, 
                 thetaL=thetaL, thetaU=thetaU,
                 theta0=theta0, cluster_method='tree', 
                 nugget=None, random_start=20,
                 nugget_estim=True, normalize=False)
                             
    optimizer = ego(dim, fitness_func, model, n_step,
                    lb=x_lb,
                    ub=x_ub, 
                    doe_size=n_init_sample,
                    solver=method,
                    verbose=False,
                    random_seed=seed)
    
    return optimizer    


optimums = {}
optimums["diffpow"] = 0
optimums["ackley"] = 0
optimums["himmelblau"] = 0
optimums["rastrigin"] = 0
optimums["schwefel"] = 0


benchmarkfunctions = {
                #"schwefel":benchmarks.schwefel,
                #"ackley":benchmarks.himmelblau,
                #"rastrigin":benchmarks.rastrigin,
                #"bohachevsky":benchmarks.bohachevsky,
                #"schaffer":benchmarks.schaffer,
                "griewank":benchmarks.griewank
                }


functionbounds = {
                  "schwefel":[-500, 500],
                  "himmelblau":[-6, 6],
                  "ackley":[-15, 30],
                  "rastrigin":[-5.12, 5.12],
                  "bohachevsky":[-100, 100],
                  "schaffer":[-100, 100],
                  "griewank":[-600, 600]}

dims = [2]
n_init_samples = [500]
n_step = 200
n_cluster = 5

methods = ['BFGS-tree', 'CMA-tree']
#methods = ['BFGS', 'CMA']

n_method = len(methods)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
runs = comm.Get_size()

try:
    os.makedirs('./data/')
except OSError:
    pass

# MPI parallel running
for n_init_sample in n_init_samples:
    
    if rank == 0:
        print 'initial samples:', n_init_sample
    
    for dim in dims:
        for func_name, func in benchmarkfunctions.iteritems():
            
            if rank == 0:
                print "function:", func_name, "dim:", dim
                
            fitness_func = lambda x: func(x)[0]
            
            for k, method in enumerate(methods):
                
                y_hist_best = np.zeros((n_step, runs))
                csv_name = './data/{}D-{}N-{}-{}.csv'.format(dim, n_init_sample, func_name, method)
                df = pd.DataFrame([], columns=['step'] + ['run{}'.format(_+1) for _ in range(runs)])
                df.to_csv(csv_name, mode='w', header=True, index=False)
                
                optimizer = create_test(dim, n_init_sample, fitness_func, 
                                        n_step, method, n_cluster, rank)
                
                for n in range(n_step):
                    xopt, fopt = optimizer.step()
                                
                    comm.Barrier()
                
                    # gather running results
                    __ = comm.gather(fopt, root=0)
                    
                    if rank == 0:
                        y_hist_best[n, :] = __
                        __mean = np.mean(__)
                        __std = np.std(__)
                        print '{} step {}:'.format(method, n+1) 
                        print 'mean: {}, std: {}'.format(__mean, __std)
                        
                        # append the new data the csv
                        df = pd.DataFrame(np.atleast_2d([n+1] + __))
                        df.to_csv(csv_name, mode='a', header=False, index=False)
                
   
