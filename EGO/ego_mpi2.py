# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 12:31:07 2014

@author: wangronin


"""

import pdb
import sys
import numpy as np
from copy import deepcopy

from pyDOE import lhs

from owck import OWCK as GaussianProcess

from scipy.stats import norm
from numpy.random import rand
from numpy import ones, array, sqrt, nonzero, min, max, sum, inf

# optimizers
from cma_es_class import cma_es
from scipy.optimize import fmin_l_bfgs_b

import random
import warnings


normcdf, normpdf = norm.cdf, norm.pdf


def ei(model, plugin=None):
    
    def __ei(X):
        
        X = np.atleast_2d(X)
        
        X = X.T if X.shape[1] != model.X.shape[1] else X
        
        n_sample = X.shape[0]
    
        if True:
            #here you did de-standardization? Not needed with OWCK
            y = model.y# * np.std(model.y) + np.mean(model.y)
        else:
            y = model.y * model.y_std + model.y_mean
        
        fmin = min(y) if plugin is None else plugin

        y_pre = []
        mse = []
        for sample in X:
            y_sample, mse_sample = model.predict(sample, eval_MSE=True)
            y_pre.append(y_sample[0])
            mse.append(mse_sample[0])
            
        y_pre = np.array(y_pre)
        mse = np.array(mse)
        y_pre = y_pre.reshape(n_sample)
        mse = mse.reshape(n_sample)

        sigma = sqrt(mse)
    
        value = (fmin - y_pre) * normcdf((fmin - y_pre) / sigma) + \
            sigma * normpdf((fmin - y_pre) / sigma)
            
        return value
        
    return __ei

def ei_dx(model, plugin=None):
    
    def __ei_dx(X):

        X = np.atleast_2d(X)
        
        X = X.T if X.shape[1] != model.X.shape[1] else X
    
        if True:
            #here you did de-standardization? Not needed with OWCK
            y = model.y# * np.std(model.y) + np.mean(model.y)
        else:
            y = model.y * model.y_std + model.y_mean
        
        fmin = min(y) if plugin is None else plugin

        y, sd2 = model.predict(X, eval_MSE=True)
        sd = np.sqrt(sd2)

        y_dx, sd2_dx = model.gradient(X)
        sd_dx = sd2_dx / (2. * sd)
        
        xcr = (fmin - y) / sd   
        xcr_prob, xcr_dens = normcdf(xcr), normpdf(xcr)
        
        grad = -y_dx * xcr_prob + sd_dx * xcr_dens
        
        return grad
        
    return __ei_dx


def pi_dx(self, x):
    pass

def minei(model, dim, plugin=None):
    
    def __minei(X):
        
        X = np.atleast_2d(X)
        
        X = X.T if X.shape[0] == dim else X
        
        n_sample = X.shape[0]
    
        if True:
            #here you did de-standardization? Not needed with OWCK
            y = model.y# * np.std(model.y) + np.mean(model.y)
        else:
            y = model.y * model.y_std + model.y_mean
        
        fmin = min(y) if plugin is None else plugin

        y_pre = []
        mse = []
        for sample in X:
            y_sample, mse_sample = model.predict(sample, eval_MSE=True)
            y_pre.append(y_sample[0])
            mse.append(mse_sample[0])
        
        #print "y",y_pre
        y_pre = np.array(y_pre)
        mse = np.array(mse)
        y_pre = y_pre.reshape(n_sample)
        mse = mse.reshape(n_sample)
        #print "y",y_pre
        #print "mse", mse

        sigma = sqrt(mse)
        
        #print (fmin - y_pre) / sigma
    
        value = (fmin - y_pre) * normcdf((fmin - y_pre) / sigma) + \
            sigma * normpdf((fmin - y_pre) / sigma)
        #print value
            
        return -1*value
        
    return __minei
    
    
def pi(model, dim, plugin=None):
    
    def __pi(X):
        
        X = np.atleast_2d(X)
        
        X = X.T if X.shape[0] == dim else X
    
        y = model.y * model.y_std + model.y_mean
        
        fmin = min(y) if plugin is None else plugin
        
        y_pre, mse = model.predict(X, eval_MSE=True)
        sigma = sqrt(mse)
        
        value = (fmin - y_pre) * normcdf((fmin - y_pre) / sigma) + \
            sigma * normpdf((fmin - y_pre) / sigma)
            
        return value
        
    return __pi
    

class ego:
    
    def __init__(self, 
                 dim, 
                 n_init_sample, 
                 fitness, 
                 n_step, 
                 lb=None,
                 ub=None, 
                 gp_model=None,
                 criterion='EI',
                 strategy='niching', 
                 sampling='LHS', 
                 parallel_eval=False,
                 is_minimize=True, 
                 useOWCK=True, 
                 method="tree", 
                 n_cluster=5, 
                 solver="CMA",
                 verbose=False,
                 random_seed=999
                 ):
        
        self.n_init_sample = n_init_sample
        self.fitness = fitness if is_minimize else lambda x: -fitness(x)
        self.dim = dim
        self.sampling = sampling
        self.is_minimize = is_minimize
        self.parallel_eval = parallel_eval
        self.useOWCK = useOWCK
        self.method = method
        self.n_cluster = n_cluster
        self.solver = solver
        self.random_seed = random_seed
        self.criterion = criterion
        
        # restart upper bound for L-BFGS-B algorithm
        # TODO: this number should be related to the dimensionality
        self.n_restarts_optimizer = 50
                
        self.n_step = n_step     # num
        self.mode = 0            # '0' mode stands for sequential evaluation
                                 # '1' mode stands for parallel evaluation
        if q is not None:
            self.q = q
            self.mode = 1
        
        x_lb = np.atleast_2d(lb)
        x_ub = np.atleast_2d(ub)
        
        self.x_lb = x_lb if x_lb.shape[0] == 1 else x_lb.T
        self.x_ub = x_ub if x_ub.shape[0] == 1 else x_ub.T

        if gp_model is None: # if no GPR model provided 
        	self.DOE()
        	self.__model_creation(X, y)
        
        
        # TODO
        # create the infill-criterion

    
        self.strategy = strategy if self.mode == 1 else None
        
        if (self.mode == 1) and (not strategy in ['CLmin', 'CLmax', 'niching']):
            raise Exception('Not supported strategy for multi-point EGO!')

        self.verbose = verbose


    def DOE(self):
    	# get old random state
        old_state1 = np.random.get_state()
        old_state2 = random.getstate()
    
        # set the new seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        if self.sampling == 'LHS':
            X = lhs(dim, samples=n_init_sample, criterion='cm') * (self.x_ub - \
                self.x_lb) + self.x_lb
        elif self.sampling == 'uniform':
            X = rand(n_init_sample, dim) * (self.x_ub - self.x_lb) + \
                self.x_lb

        # restore the random state
        np.random.set_state(old_state1)
        random.setstate(old_state2)
        
        y = self.__parallel_evaluation(X) if self.parallel_eval else \
            array([self.fitness(x) for x in X])
        
        # create the meta model
        self.X, self.y = X, y
            
    
    def __model_creation(self, X, y):
        
        # bounds of hyper-parameters for optimization (not needed for normalization)
        thetaL = 1e-3 * ones(self.dim) * (self.x_ub - self.x_lb) 
        thetaU = 10 * ones(self.dim) * (self.x_ub - self.x_lb) 
        
        # initial search point for hyper-parameters
        theta0 = rand(1, self.dim) * (thetaU - thetaL) + thetaL
        if self.useOWCK:
            self.model = OWCK(regr='constant', corr='matern', 
                 n_cluster=self.n_cluster, min_leaf_size=0, 
                 cluster_method=self.method, overlap=0.0, beta0=None, 
                 storage_mode='full', verbose=False, theta0=theta0, thetaL=thetaL, 
                 thetaU=thetaU, optimizer='fmin_cobyla', random_start=1, nugget=0.000001,
                 normalize=False,
                 is_parallel=False)
        else:
            self.model = GaussianProcess(regr='constant', 
                                         corr='matern', 
                                         theta0=theta0, 
                                         thetaL=thetaL, 
                                         normalize=False,
                                         nugget=0.000001,
                                         thetaU=thetaU)
        # model fitting
        self.model.fit(X, y)
            
            
    def __parallel_evaluation(self, X):
        
        from mpi4py import MPI
        from pandas import DataFrame
        
        comm = MPI.COMM_WORLD
        
        n_procs, _ = X.shape
        
        # Spawning processes to test kriging mixture
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=['ego_evaluation.py'], 
                                   maxprocs=n_procs)
    
        # scatter the models and data
        comm.bcast(self.fitness, root=MPI.ROOT)
        comm.scatter([(k, X[k, :]) \
            for k in range(n_procs)], root=MPI.ROOT)
        
        # Synchronization while the children process are performing 
        # heavy computations...
        comm.Barrier()
            
        # Gether the fitted model from the childrenn process
        # Note that 'None' is only valid in master-slave working mode
        result = comm.gather(None, root=MPI.ROOT)
        
        comm.Disconnect()
        
        # register the measures
        data = DataFrame([[d['index'], d['y']] \
            for d in result], columns=['index', 'y'])
        data.sort('index', inplace=True)
        
        return array(data['y'])
      
        
    def optimize(self):
        y_hist = []
        if self.mode == 0:
            best_x, best_y, new_X, new_y, y_hist = self.nstep()
        elif self.mode == 1:
            best_x, best_y, new_X, new_y = self.nstep_parallel()
        
        return best_x, best_y, new_X, new_y, y_hist
        
            
    def nstep(self):
        y_hist = []
        for i in range(self.n_step):
            new_x = self.max_ei(self.solver, min(self.y))
            new_x = self.__remove_duplicate(new_x)
            
            if len(new_x) == 0:
                y_hist.append( min(self.y))
                continue
            
            new_y = []
            for x in new_x:
                new_y.append(self.fitness(x)[0])
            new_y = np.array(new_y)
            #print "found new x", new_x
            #print "with value:",new_y
            
            self.X = np.r_[self.X, new_x]
            self.y = np.r_[self.y, new_y]

            y_hist.append(min(self.y))
            
            if self.useOWCK and self.method=="tree":
                self.model.updateModel(np.atleast_2d(new_x),new_y)
                #print(len(self.model.y))
            else:
                self.__model_creation(self.X, self.y)
            
            #print ('step {}, best fitness {}'.format(i+1, min(self.y)) )
        
        # set up the best solution found
        idx = nonzero(self.y == min(self.y))[0]
        best_x = self.X[idx, :]
        best_y = self.y[idx]
        best_y = best_y if self.is_minimize else -best_y
        
        return best_x, best_y, self.X[self.n_init_sample:, :], \
            self.y[self.n_init_sample:], y_hist
            
        
    def nstep_parallel(self):
        
        X_per_step = []
        y_per_step = []
        for i in range(self.n_step):
            
            if self.strategy == 'CL':
                new_x, _ = self.__parallel_CL()
            
            elif self.strategy == 'niching':
                new_x, _ = self.__parallel_niching()
            
            new_x = self.__remove_duplicate(new_x)
            if new_x is not []:
                new_y = self.__parallel_evaluation(new_x) if self.parallel_eval \
                    else array([self.fitness(x) for x in new_x])
            else:
                new_y = []
            
            if len(new_y) == 0:
                continue
            
            X_per_step.append(new_x)
            y_per_step.append(new_y)
            
            self.X = np.r_[self.X, new_x]
            self.y = np.r_[self.y, new_y]
            
            self.__model_creation(self.X, self.y)
            
            print('step {}, best fitness {}'.format(i+1, min(new_y)))
        
        # set up the best solution found
        idx = nonzero(self.y == min(self.y))
        best_x = self.X[idx, :]
        best_y = self.y[idx]
        best_y = best_y if self.is_minimize else -best_y
        
        return best_x, best_y, X_per_step, y_per_step
            
    def __remove_duplicate(self, new_x):
        
        # TODO: show a warning here
        new_x = np.atleast_2d(new_x)
        samples = []
        for x in new_x:
            if not any(sum(np.isclose(self.X, x), axis=1)):
                samples.append(x)
        
        return array(samples)
        
    def __parallel_CL():
        pass
        
    def __parallel_niching(self):
        
        from niching_dr1 import niching_dr1
        
        x_lb, x_ub = self.x_lb, self.x_ub
        
        q = self.q
        q_eff = q + 3
        rho = sqrt(sum((x_ub - x_lb) ** 2, axis=1)) / 2. / q**(1./self.dim)
        kappa = 1e2
        sigma0 = 0.1 * max(x_ub - x_lb)
        eval_budget = q_eff*1e3
        fit_ei = ei(self.model, self.dim)
        
        peaks, fit = niching_dr1(fit_ei, self.dim, x_lb.T, x_ub.T, q, q_eff, rho, 
                            kappa, sigma0, eval_budget, is_minimize=False)
        
        return peaks.T, fit
        
    
    def __get_tree_boundary(self):
        """
        Get the variable boundary for each leaf node
        """
        
        def recurse(tree, node_id, decision_path, bounds):
            """
            Traverse the decision tree in preoder to retrieve the 
            decision path for leaf nodes
            """
            decision_path_ = deepcopy(decision_path)
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            split_feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            
            if split_feature == -2: # leaf node
                leaf_node_path[node_id] = decision_path_
                leaf_bounds[node_id] = bounds
                return
            else:
                decision_path_.append(node_id) # pre-order visit
                __ = bounds[split_feature]
                
                # proceed to the left child
                bounds_left = deepcopy(bounds)
                bounds_left[split_feature] = update_bounds(__, leq=threshold)
                recurse(tree, left_child, decision_path_, bounds_left)
                
                # proceed to the right child
                bounds_right = deepcopy(bounds)
                bounds_right[split_feature] = update_bounds(__, geq=threshold)
                recurse(tree, right_child, decision_path_, bounds_right)
        
        def update_bounds(bounds, leq=None, geq=None):
            lb, ub = bounds
            if leq is not None:
                if leq <= lb:
                    raise Exception
                ub = leq if leq < ub else ub
                
            if geq is not None:
                if geq >= ub:
                    raise Exception
                lb = geq if geq > lb else lb
                
            return [lb, ub]
        
        leaf_node_path = {}  
        leaf_bounds = {}
        tree = self.model.clusterer.tree_
        
        recurse(tree, 0, [], 
                {i: [self.x_lb[0, i], self.x_ub[0, i]] for i in range(self.dim)})
        
        return leaf_bounds
        
        
    def max_ei(self, solver='BB', plugin=None):
        
        if plugin is None:
            plugin = min(self.y)
        
        # CMA-ES with Decision tree search
        if self.solver == 'CMA-tree':
            leaf_bounds = self.__get_tree_boundary()
            total_budget = 1e5*self.dim
            total_measure = np.prod(self.x_ub - self.x_lb)
            
            # TODO: this part should be parallelizable
            fopt = -inf
            for leaf_node, bounds in leaf_bounds.iteritems():
                
                x_lb, x_ub = zip(*[bounds[i] for i in bounds.keys()])
                x_lb = np.atleast_2d(x_lb)
                x_ub = np.atleast_2d(x_ub)
                
                measure = np.prod(x_ub - x_lb)
                eval_budget = int(measure / total_measure * total_budget)
                
                # Algorithm parameters
                opt = {
                       'sigma_init': 0.25 * np.max(x_lb - x_ub),
                       'eval_budget': eval_budget, 
                       'f_target': -inf, 
                       'lb': x_lb.T, 
                       'ub': x_ub.T
                       }
                    
                init_wcm = rand(1, self.dim) * (x_ub - x_lb) + x_lb
                fitnessfunc = ei(self.model)
                
                optimizer = cma_es(self.dim, init_wcm, fitnessfunc, opt, 
                                   is_minimize=False)
                xopt_, fopt_, evalcount, _ = optimizer.optimize()
                
                if fopt_ > fopt:
                    fopt = fopt_
                    xopt = xopt_.T

            return xopt

        #standard scipy optimizers:
        if solver == "diff":
            from scipy.optimize import differential_evolution
            fitnessfunc = minei(self.model, self.dim, plugin)
            bounds = []
            for i in range(len(self.x_lb[0])):
                bounds.append([self.x_lb[0][i],self.x_ub[0][i]])
            #print bounds
            testfopt = fitnessfunc(np.ones(self.dim)*0)
            result = differential_evolution(fitnessfunc, bounds, maxiter=10000)
            #print "x:",result.x 
            print "fopt", result.fun
            print "testfopt", testfopt
            print result.x
            return [result.x]
            
        elif solver == "basinhopping":

            from scipy.optimize import basinhopping
            fitnessfunc = minei(self.model, self.dim, plugin)
            bounds = []
            for i in range(len(self.x_lb[0])):
                bounds.append([self.x_lb[0][i],self.x_ub[0][i]])
            #print bounds
            testfopt = fitnessfunc(np.ones(self.dim)*0)
            minimizer_kwargs = {"method": "BFGS"}
            ri = random.randint(0, len(self.X))
            result = basinhopping(fitnessfunc,self.X[ri],  minimizer_kwargs=minimizer_kwargs, niter=5000)
            #print "x:",result.x 
            print "fopt", result.fun
            print "testfopt", testfopt
            print result.x
            return [result.x]
            
        elif solver =="pso":
            from pyswarm import pso
            fitnessfunc = minei(self.model, self.dim, plugin)
            testfopt = fitnessfunc(np.ones(self.dim)*0)
            xopt, fopt = pso(fitnessfunc, self.x_lb, self.x_ub)
            print "fopt", fopt
            print "testfopt",testfopt
            print xopt
            return [xopt]
        
        elif solver == 'BB':   # Branch and Bound solver
            # TODO: implement this for Gaussian kernel
            pass
        elif solver == 'genoud':    # Genetic optimization with derivatives
            # TODO: somehow get it working from R package
            pass
        elif solver == 'BFGS':      # quasi-newton method with restarts
                    
            EI, EI_dx = ei(self.model), ei_dx(self.model)
            def obj_func(x): return -EI(x), -EI_dx(x)
            
            fopt = np.inf
            eval_budget = 1000*self.dim
            bounds = np.c_[self.x_lb.T, self.x_ub.T]
            
            if not np.isfinite(bounds).all() and self.n_restarts_optimizer > 1:
                raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
            
            # L-BFGS-B algorithm with restarts
            for iteration in range(self.n_restarts_optimizer):
                
                x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
                
                xopt_, fopt_, convergence_dict = \
                    fmin_l_bfgs_b(obj_func, x0, pgtol=1e-30, factr=1e5,
                                  bounds=bounds, maxfun=eval_budget)
                
                if convergence_dict["warnflag"] != 0:
                    warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                          " state: %s" % convergence_dict)
                          
                xopt_list.append(xopt_)
                
                if fopt_ < fopt:
                    if self.verbose:
                        print 'iteration: ', iteration+1, convergence_dict['funcalls']
                        
                    xopt, fopt = xopt_, fopt_
                
                eval_budget -= convergence_dict['funcalls']
                
                if eval_budget <= 0:
                    break
            
        elif solver == 'CMA':        # CMA-ES solver
            
            # TODO: use BI-POP/ IPOP-CMA-ES here
            # Algorithm parameters
            opt = { \
                'sigma_init': 0.25 * np.max(self.x_lb - self.x_ub), 
                'eval_budget': 1e5*self.dim, 
                'f_target': -inf, 
                'lb': self.x_lb.T, 
                'ub': self.x_ub.T \
                }
                
            init_wcm = rand(1, self.dim) * (self.x_ub - self.x_lb) + \
                self.x_lb
            
            fitnessfunc = ei(self.model)
            testfopt = fitnessfunc(np.ones(self.dim)*0)
            optimizer = cma_es(self.dim, init_wcm, fitnessfunc, opt, 
                               is_minimize=False) #maximize ei
            xopt, fopt, evalcount, _ = optimizer.optimize()
            xopt = xopt.T

        return xopt
            

xopt_list = []
if __name__ == '__main__':
     
    
    from utils import plot_contour_gradient
    import matplotlib.pyplot as plt
    
#    plt.style.use('ggplot')
    np.random.seed(370)
    
    fig_width = 22
    fig_height = fig_width * 9 / 16
    
    # To verify the corr gradient computation
    x_lb = [-5, -5]
    x_ub = [5, 5]
    
    n_sample = 10
    
    def fitness(X):
        X = np.atleast_2d(X)
        return np.sin(np.sum(X, axis=1) ** 2.0)
    
    X = np.random.rand(n_sample, 2) * n_sample - 5
    y = fitness(X)
    
    model = GaussianProcess(corr='matern', cluster_method='tree',
                            n_cluster=3, theta0=[1e-1, 5*1e-1], normalize=False)
    model.fit(X, y)
    
    # Plot EI and PI landscape 
    fig0, ax0 = plt.subplots(1, 1, figsize=(fig_width, fig_height), 
                                    subplot_kw={'aspect':'equal'}, dpi=100)
                        
    ax0.set_xlim([x_lb[0], x_ub[0]])
    ax0.set_ylim([x_lb[1], x_ub[1]])
        
    f = ei(model)
    grad = ei_dx(model)
    
    f_data = np.load('fitness.npy')
    grad_data = np.load('grad.npy')
        
    plot_contour_gradient(ax0, f, grad, x_lb, x_ub, title='Expected Improvement', 
                          n_per_axis=100)
    
    # diagnostic comparisons of EI maximization methods
    color = ['k', 'g']
    for i, method in enumerate(['CMA-tree', 'BFGS']):
        optimizer = ego(2, 10, 
                        fitness, 1,
                        lb=x_lb, 
                        ub=x_ub, 
                        useOWCK=False, 
                        n_cluster=1,
                        solver=method,
                        verbose=True,
                        random_seed=0)
                    
        optimizer.model = model
        optimizer.X = model.X
        optimizer.y = model.y
    
        _, __, new_X, new_y, ___ = optimizer.optimize()
        
        ax0.plot(new_X[-1, 0], new_X[-1, 1], ls='none', marker='.', ms=15, 
                 mfc=color[i], mec='none', alpha=0.7)
                 
        if len(xopt_list) != 0:
            __ = np.array(xopt_list)
            ax0.plot(__[:, 0], __[:, 1], ls='none', marker='.', ms=15, 
                 mfc=color[i], mec='none', alpha=0.7, clip_on=False)
    
    
    plt.show()


    




