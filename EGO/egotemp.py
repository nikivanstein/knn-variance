# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 12:31:07 2014

@author: wangronin


"""

import pdb
import numpy as np
from copy import deepcopy

from pyDOE import lhs



from scipy.stats import norm
from numpy.random import rand
from numpy import ones, array, sqrt, nonzero, inf

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
        
        fmin = np.min(y) if plugin is None else plugin

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
        
        fmin = np.min(y) if plugin is None else plugin

        y, sd2 = model.predict(X, eval_MSE=True)
        sd = np.sqrt(sd2)

        y_dx, sd2_dx = model.gradient(X)
        sd_dx = sd2_dx / (2. * sd)
        
        xcr = (fmin - y) / sd   
        xcr_prob, xcr_dens = normcdf(xcr), normpdf(xcr)
        
        grad = -y_dx * xcr_prob + sd_dx * xcr_dens
        
        return grad
        
    return __ei_dx


def pi(model, dim, plugin=None):
    
    def __pi(X):
        
        X = np.atleast_2d(X)
        
        X = X.T if X.shape[0] == dim else X
    
        y = model.y * model.y_std + model.y_mean
        
        fmin = np.min(y) if plugin is None else plugin
        
        y_pre, mse = model.predict(X, eval_MSE=True)
        sigma = sqrt(mse)
        
        value = (fmin - y_pre) * normcdf((fmin - y_pre) / sigma) + \
            sigma * normpdf((fmin - y_pre) / sigma)
            
        return value
        
    return __pi
    

def pi_dx(self, x):
    pass
    

class ego:
    
    def __init__(self, dim, fitness, model, n_step, lb, ub,
                 doe_size=None,
                 doe_method='LHS',
                 criterion='EI',
                 is_minimize=True,
                 solver='BFGS',
                 verbose=False,
                 random_seed=999
                 ):
        
        self.is_minimize = is_minimize
        self.verbose = verbose
        self.dim = int(dim)
        self.n_step = int(n_step)     # number of steps
        self.random_seed = random_seed
        
        # set the new seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        if criterion not in ['EI', 'PI']:
            raise Exception('Unsupported in-fill criterion!')
        else:
            self.criterion = criterion

        # TODO: callable solvers
        if solver not in ['CMA', 'BFGS', 'CMA-cluster', 'BFGS-cluster']:
        	raise Exception('Unsupported solver for in-fill criterion!')
        else:
        	self.solver = solver

        if hasattr(fitness, '__call__'):
            self.fitness = fitness if is_minimize else lambda x: -fitness(x)
        else:
            raise Exception('fitness function is not callable!')

        if len(lb) != self.dim or len(ub) != self.dim:
            raise Exception('Length of bounds does not match dimension!')

        x_lb = np.atleast_2d(lb)
        x_ub = np.atleast_2d(ub)
        
        self.x_lb = x_lb if x_lb.shape[0] == 1 else x_lb.T
        self.x_ub = x_ub if x_ub.shape[0] == 1 else x_ub.T
        
        self.model = deepcopy(model)
        if hasattr(self.model, 'X'):    # given fitted Kriging model
            if not isinstance(model, GaussianProcess):
                raise Exception('Unsupported Model class!')
                
               # make a copy, safe :)
            self.X = self.model.X
            self.y = self.model.y 
            self.doe_size = self.X.shape[0]

            if self.y.ndim != 1:
                self.y = self.y.flatten()

        else:                      # otherwise the model will be fitted
            self.doe_method = doe_method
            self.doe_size = int(doe_size)
            self.X, self.y = self.__DOE()

            if self.verbose:
                print 'creating the GPR model on DOE samples...'
            self.__model_fitting(self.X, self.y)
        
        # restart upper bound for L-BFGS-B algorithm
        # TODO: this number should be related to the dimensionality
        self.n_restarts_optimizer = int(1e2)


    def evaluation(self, X):
        n_sample, _ = X.shape
        y = array([self.fitness(x) for x in X])

        if y.ndim != 1:
            y = y.flatten()

        return y
            

    def __DOE(self):
    	"""
    	Design of Experiments...
    	"""
    	# get old random state
        # old_state1 = np.random.get_state()
        # old_state2 = random.getstate()
    
        if self.doe_method == 'LHS':
            X = lhs(self.dim, samples=self.doe_size, criterion='cm') \
            	* (self.x_ub - self.x_lb) + self.x_lb
        elif self.doe_method == 'uniform':
            X = rand(self.doe_size, self.dim) * (self.x_ub - self.x_lb) \
            	+ self.x_lb

        # restore the random state
        # np.random.set_state(old_state1)
        # random.setstate(old_state2)
        
        y = self.evaluation(X)
        
        return X, y
            
    
    def __model_fitting(self, X, y):
        
        # bounds of hyper-parameters for optimization (not needed for normalization)
        thetaL = 1e-3 * ones(self.dim) * (self.x_ub - self.x_lb) 
        thetaU = 10 * ones(self.dim) * (self.x_ub - self.x_lb) 
        
        # initial search point for hyper-parameters
        theta0 = rand(1, self.dim) * (thetaU - thetaL) + thetaL

        self.model.thetaL = thetaL
        self.model.thetaU = thetaU
        self.model.theta0 = theta0

        # model fitting
        self.model.fit(X, y)
            
        
    def optimize(self):
        # TODO: add stop criteria...
        best_x, best_y, new_X, new_y, y_hist = self.nstep()
        
        return best_x, best_y, new_X, new_y, y_hist
        
            
    def nstep(self):
        y_hist = []
        for i in range(self.n_step):
            new_x = self.max_ei(self.solver, np.min(self.y))
            new_x = self.__remove_duplicate(new_x)
            
            if len(new_x) == 0:
            	warnings.warn('duplicated solution found!')
                y_hist.append(np.min(self.y))
                self.__model_fitting(self.X, self.y)
                #continue
            else:
                new_y = self.evaluation(new_x)
                self.X = np.r_[self.X, new_x]
                self.y = np.r_[self.y, new_y]

                y_hist.append(np.min(self.y))
                
                self.__model_fitting(self.X, self.y)
        
        # set up the best solution found
        idx = nonzero(self.y == np.min(self.y))[0]
        best_x = self.X[idx, :]
        best_y = self.y[idx]
        best_y = best_y if self.is_minimize else -best_y
        
        return best_x, best_y, self.X[self.doe_size:, :], \
            self.y[self.doe_size:], y_hist
            
            
    def __remove_duplicate(self, new_x):
        
        # TODO: show a warning here
        new_x = np.atleast_2d(new_x)
        samples = []
        for x in new_x:
            if not any(np.sum(np.isclose(self.X, x), axis=1)):
                samples.append(x)
        
        return array(samples)
        
    
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
        
        
    def max_ei(self, solver='BFGS', plugin=None):
        
        if plugin is None:
            plugin = np.min(self.y)

        #standard scipy optimizers:
        if solver == "diff":
            from scipy.optimize import differential_evolution
            def fitnessfunc(x): return -EI(x)
                
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
                
                if convergence_dict["warnflag"] != 0 and self.verbose:
                    warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                          " state: %s" % convergence_dict)
                          
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
                'sigma_init': 0.25 * np.max(self.x_ub - self.x_lb), 
                'eval_budget': 1e4*self.dim, 
                'f_target': inf, 
                'lb': self.x_lb.T, 
                'ub': self.x_ub.T \
                }
                
            init_wcm = rand(1, self.dim) * (self.x_ub - self.x_lb) + \
                self.x_lb

            #print "init point",init_wcm
            
            fitnessfunc = ei(self.model)
            optimizer = cma_es(self.dim, init_wcm, fitnessfunc, opt, 
                               is_minimize=False)
            xopt, fopt, evalcount, _ = optimizer.optimize()
            #print "xopt",xopt
            #print "fopt",fopt
            xopt = xopt.T
            
        # CMA-ES with Decision tree search
        elif self.solver == 'CMA-tree':
            leaf_bounds = self.__get_tree_boundary()
            total_budget = 1e4*self.dim
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
                       'sigma_init': 0.25 * np.max(x_ub - x_lb),
                       'eval_budget': eval_budget, 
                       'f_target': inf, 
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

        elif solver == 'BB':   # Branch and Bound solver
            # TODO: implement this for Gaussian kernel
            pass
        elif solver == 'genoud':    # Genetic optimization with derivatives
            # TODO: somehow get it working from R package
            pass

        return xopt
