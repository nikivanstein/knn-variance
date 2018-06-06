# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 15:06:40 2014

@author: wangronin
"""

#Dynamic ES Niching with (1,lambda)-DR1.
# -----------------------------------------------------------------------------------
# Niching DR1: Dynamic Niching with Derandomized Evolution Strategy
# for nonlinear def multimodal minimization. To be used under the
# terms of the GNU General Public License:
# http://www.gnu.org/copyleft/gpl.html
# DR1 implementation is based on Ostermeier, Gawelczyk, Hansen: A derandomized approach 
# to self adaptation of evolution strategies (1993).
#
# Author: Ofer M. Shir, 2006. e-mail: oshir@liacs.nl
# http://www.liacs.nl/~oshir/
# -----------------------------------------------------------------------------------

import pdb
import numpy as np
from numpy.random import randn, rand
from numpy import argsort, zeros, ones, nonzero, array, sqrt, exp,  \
    r_, size, ceil, tile, pi, sum, any

alpha = None
beta = None
beta_s = None

def niching_dr1(fitnessfunc, dim, x_lb, x_ub, q, q_eff, rho, kappa,
                co_sigma, eval_budget, is_minimize=True):

    # Strategy parameter setting: Selection
    _lambda = 10  
    
    # Strategy parameter setting: Adaptation
    global alpha, beta, beta_s
    alpha = 1.4
    beta = 1.0/sqrt(dim)
    beta_s = 1.0/dim
    
    # Data-structures
    p = q_eff - q
    pop_size = q_eff*_lambda
    X = (x_ub-x_lb) * rand(dim, q_eff) + x_lb #decision parameters to be optimized.
    Y = zeros((dim, pop_size)) #temporary DB for offspring.
    P = zeros(pop_size, dtype=np.int) #Parents indices
    
    # Initialize dynamic (internal) strategy parameters and constants
    delta = array([init_dr1(dim, co_sigma) for i in range(q_eff)])
    
    gen = 0
    global_eval = 0
    fitness = zeros(pop_size)
    
    max_gen = ceil(eval_budget/(pop_size))
    stat = zeros(max_gen)
    mpr_q = zeros((q, max_gen))
    
    # -------------------- Generation Loop --------------------------------
    while global_eval < eval_budget:
        
        z = randn(dim, pop_size)
        xi = alpha*(rand(pop_size) >= 0.5) 
        xi[xi==0] = 1.0/alpha
        
        for k in range(q_eff):
            Y[:, _lambda*k:_lambda*(k+1)] = X[:, k].reshape(-1, 1) + xi[k] * delta[k, -1] *\
                delta[k, 0:-1].reshape(-1, 1) * z[:, _lambda*k:_lambda*(k+1)]
            P[_lambda*k:_lambda*(k+1)] = k 
        
        #Periodic Boundary Conditions - Let us keep X in the interval [x_lb,x_ub]:
        if any(Y - x_lb < 0) or any(Y - x_ub > 0):
            Y[:, :] = Y[:, :] * (Y - x_lb >= 0) * (Y - x_ub <= 0) + \
                 tile(x_lb, (1, pop_size)) * (Y - x_lb < 0) + tile(x_ub, (1, pop_size)) * (Y - x_ub > 0)
         
        # Fitness evaluation + sorting
        fitness[:] = fitnessfunc(Y)
        global_eval += size(Y, axis=1)
        
        rank = argsort(fitness) if is_minimize else argsort(fitness)[::-1]
        fitness[:] = fitness[rank]
        Y[:, :] = Y[:, rank] # Decision+Strategy parameters are now sorted!
        z[:, :] = z[:, rank] 
        xi[:] = xi[rank] 
        P[:] = P[rank]
    
        stat[gen] = fitness[0]
        
        #Dynamic Peak Identification
        dps, _ = dpi(Y, pop_size, q, rho)
        
        #(1,lambda) Selection for each niche
        new_delta = zeros((q_eff, dim + 1))
        for i in range(q):
            j = dps[i]
            if j!= -1:
                parent = P[j] #the original parent!
                X[:, i] = Y[:, j]
                new_delta[i, :] = dr1_adapt(delta[parent, :], z[:, j], xi[j], dim)
            else:
                X[:, i] = ((x_ub-x_lb) * rand(dim, 1) + x_lb).T
                new_delta[i, :] = init_dr1(dim, co_sigma)
        
#        for i in range(q):
#            if all(np.isclose(X[:, i], x_ub[:, 0])) or \
#                all(np.isclose(X[:, i], x_lb[:, 0])):
#                X[:, i] = ((x_ub-x_lb) * rand(dim, 1) + x_lb).T
#                new_delta[i, :] = init_dr1(dim, co_sigma)
            
        if gen % kappa == 0:
            X[:, q:] = (x_ub-x_lb) * rand(dim, p) + x_lb
            new_delta[q:, :] = array([init_dr1(dim, co_sigma) for i in range(p)])
            
        delta = new_delta
        MX = fitnessfunc(X[:, :q])
    
        # Output
    #     if (mod(gen,out)==0)
    #         disp([num2str(gen) ': ' num2str(MX(:,:))])
    
        mpr_q[:, gen] = MX #1./(1+abs(MX(:,:)'))
        gen += 1
    
    peaks = X[:, :q]
    return peaks, MX
# disp([num2str(gen) ': ' num2str(MX(:,:))])


#--------------------------------------------------------------------------
def init_dr1(dim, co_sigma):
    return r_[ones(dim), co_sigma]


#--------------------------------------------------------------------------
def dr1_adapt(delta, arz, xi, dim):
    
    global beta_s, beta
    new_delta = zeros(dim+1)
    new_delta[:dim] = delta[:dim] * (exp(abs(arz) - sqrt(2.0/pi)) ** beta_s)
    new_delta[-1] = delta[-1] * (xi ** beta)
    return new_delta


#--------------------------------------------------------------------------
def dpi(Y, pop_size, q, rho):
    
    dps = -1 * ones(q, dtype=np.int) #Dynamic Peak Set.
    pop_niche = -1 * ones(pop_size, dtype=np.int) #The classification of each individual to a niche zero is "non-peak" domain.
    num_peaks = 1
    niche_count = zeros(q, dtype=np.int)
    
    dps[0] = 0
    niche_count[0] = 1
    pop_niche[0] = 0
    
    for k in range(1, pop_size):
        d_pi = Y[:, k].reshape(-1, 1) - Y[:, dps[dps!=-1]]
        tmp = sum(d_pi ** 2., axis=0) < rho**2.
        if any(tmp):
            j = nonzero(tmp)[0][0]
            niche_count[j] += 1
            pop_niche[k] = j
        elif num_peaks < q:
            dps[num_peaks] = k
            niche_count[num_peaks] = 1
            pop_niche[k] = num_peaks
            num_peaks += 1
    
    pop_niche[pop_niche==-1] = q+1
    
    return dps, pop_niche

#--------------------------------------------------------------------------
