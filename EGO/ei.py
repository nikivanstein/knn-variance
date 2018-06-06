# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:47:30 2014

@author: wangronin
"""

import pdb
import numpy as np

from scipy.stats import norm
from scipy.linalg import solve_triangular
from numpy.linalg import inv, eigh
from numpy.random import rand
from numpy import size, min, inf, sqrt, dot, atleast_2d, any
from scipy.optimize import fmin_slsqp, fmin_cobyla, fmin_tnc

from sklearn.metrics.pairwise import manhattan_distances
from openopt import NLP

def ei_generator(dmodel):
    
    def ei(x):
        
        normcdf, normpdf = norm.cdf, norm.pdf
        
        _, n_features = dmodel.X.shape
        
        if x.shape[0] == n_features:
            x = x.T
    
        y = dmodel.y * dmodel.y_std + dmodel.y_mean
        fmin = min(y)
        
        y_pre, mse = dmodel.predict(x, eval_MSE=True)
            
        sd = sqrt(mse)
        res = (fmin - y_pre) * normcdf((fmin - y_pre) / sd) + sd * normpdf((fmin - y_pre) / sd)
        
        return -res
        
    return ei
    
def upper_bound_mse(model, lower, upper):
    
    n_sample, n_feature = model.X.shape
    x_lb = atleast_2d(lower)
    x_ub = atleast_2d(upper)
    
    x_lb = x_lb.T if size(x_lb, 1) != n_feature else x_lb
    x_ub = x_ub.T if size(x_ub, 1) != n_feature else x_ub
    
    x0 = ((x_ub - x_lb) * rand(1, n_feature) + x_lb)[0]
    
    obj = objective_mse_wrapper(model)
#    x_opt = fmin_slsqp(obj, x0, bounds=zip(x_lb[0], x_ub[0]), iter=1e4, iprint=-1)
    
#    # openopt optimizers
    p = NLP(obj, x0, lb=x_lb[0], ub=x_ub[0], iprint=1e6)
    r = p.solve('ralg')
    x_opt = r.xf
    
#    if any(x_opt < x_lb) or any(x_opt > x_ub):
#        pdb.set_trace()
    res = -obj(x_opt)
    
    return res


# Calculate the mse of the kringing model
# to facilitate the upper bound computation 
# of mse, following D.R.Jones paper
def objective_mse_wrapper(model):
    def objective_mse(x):
        
        C = model.C
        Ft = model.Ft
        G = model.G
        X = model.X
        y = model.y
        theta_ = model.theta_
        sigma2 = model.sigma2
        
        n_samples, n_features = X.shape
        n_samples_y, n_targets = y.shape
        
        x = atleast_2d(x)
        x = x.T if x.shape[1] != n_features else x
        n_eval, _ = x.shape
        
        X = (X - model.X_mean) / model.X_std
        
#        C_inv = inv(C)
#        hessian = 2 * sigma2 * (dot(C_inv.T, inv) - dot(dot(dot(C_inv.T, Ft), inv(dot(Ft.T, Ft))), dot(Ft.T, C_inv)))
#        e_value, _ = eigh(hessian)
#        lambda_min = min(e_value)
#        alpha = max([0, -lambda_min/2])
        
        dx = manhattan_distances(x, Y=X, sum_over_features=False)
        # Get regression function and correlation
        f = model.regr(x)
        r = model.corr(theta_, dx).reshape(n_eval, n_samples)
        
        rt = solve_triangular(C, r.T, lower=True)
        u = solve_triangular(G.T, np.dot(Ft.T, rt) - f.T)
        
        mse = -sigma2 *(1. - (rt ** 2.).sum(axis=0) + (u ** 2.).sum(axis=0))
        return mse
        
    return objective_mse

def lower_bound_predictor(model, lower, upper):

    
    n_sample, n_feature = model.X.shape
    
    theta = model.theta_
    ph = 2
    beta = model.beta
    gamma = model.gamma
    X = model.X
    
    x_lb = atleast_2d(lower)
    x_ub = atleast_2d(upper)
    
    x_lb = x_lb.T if size(x_lb, 1) != n_feature else x_lb
    x_ub = x_ub.T if size(x_ub, 1) != n_feature else x_ub

    x_lb = (x_lb - model.X_mean) / model.X_std
    x_ub = (x_ub - model.X_mean) / model.X_std
    
    
#    a = zeros(1, m) b = zeros(1, m)
#    midpoint = (z_min + z_max)' / 2
#    for i = 1:m
#        if gamma(i) >= 0 
#            # tangent line relaxation
#            b(i) = -gamma(i) * exp(-midpoint(i))
#            a(i) = gamma(i) * exp(-midpoint(i)) - b(i) * midpoint(i)
#        else
#            # chord relaxation
#            b(i) = (gamma(i) * exp(-z_max(i)) - gamma(i) * exp(-z_min(i))) / (z_max(i)-z_min(i))
#            a(i) = gamma(i) * exp(-z_min(i)) - b(i) * z_min(i)
#        
    
    x0 = ((x_ub - x_lb) * rand(1, n_feature) + x_lb)[0].tolist()  # Make a starting guess at the solution
    obj = model.predict
#    x_opt = fmin_slsqp(obj, x0, bounds=zip(x_lb[0], x_ub[0]), iter=1e4, iprint=-1)
    p = NLP(obj, x0, lb=x_lb[0], ub=x_ub[0], iprint=1e6)
    r = p.solve('ralg')
    x_opt = r.xf
    
#    if any(x_opt < x_lb) or any(x_opt > x_ub):
#        pdb.set_trace()
    res = obj(x_opt)
    
    return res
  
#def z_bound(model, x_lb, x_ub):
#    S = model.S
#    
#    [m, n] = size(model.S)
#    dx1 = repmat(upper, m, 1) - S
#    dx2 = repmat(lower, m, 1) - S
#    index = dx1 >= 0 & dx2 <= 0
#    
#    r_lb = zeros(size(index))
#    r_ub = zeros(size(index))
#    for i = 1 : m
#        if index(i, 1) == 1
#            r_ub(i, 1) = S(i, 1)
#            if abs(dx1(i, 1)) >= abs(dx2(i, 1))
#                r_lb(i, 1) = upper(1)
#            else
#                r_lb(i, 1) = lower(1)
#            
#        else
#            if dx1(i, 1) >= 0
#                r_lb(i, 1) = upper(1)
#                r_ub(i, 1) = lower(1)
#            else
#                r_lb(i, 1) = lower(1)
#                r_ub(i, 1) = upper(1)
#            
#        
#        if index(i, 2) == 1
#            r_ub(i, 2) = S(i, 2)
#            if abs(dx1(i, 2)) >= abs(dx2(i, 2))
#                r_lb(i, 2) = upper(2)
#            else
#                r_lb(i, 2) = lower(2)
#            
#        else
#            if dx1(i, 2) >= 0
#                r_lb(i, 2) = upper(2)
#                r_ub(i, 2) = lower(2)
#            else
#                r_lb(i, 2) = lower(2)
#                r_ub(i, 2) = upper(2)
            
        
    
            


#def res = objective_preditor(x)
#    
#    global model
#    
#    S = model.S
#    beta = model.beta gamma = model.gamma
#    
#    [m, n] = size(model.S)
#    dx = repmat(x, m, 1) - S
#    [r, dr] = feval(model.corr, model.theta, dx)
#    f = feval(model.regr, x)
#    res = f * beta + (gamma*r).'
#   
    

