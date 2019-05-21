# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:14:49 2019

@author: jchaconhurtado

Implementation of Lahtinen2017 (RPM)

"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import plot_help
import utils

budget_limit = 45.0
# target to reduce water by 50%
# Objectives reduce Cc impacts, financial effects, eefects in the water system

# 9 action candidates
#        Phosporous, Nitrogen,     Climate,      Savings,      Cost,     Water
action = [
        [[0.9, 1.1], [0.09, 0.11], [0.0, 0.0],   [1.8, 2.2],   1.0,  0.03, ], #1
        [[1.1, 1.3], [0.09, 0.11], [0.0, 0.0],   [1.8, 2.2],   1.0,  0.07, ], #2
        [[1.3, 1.7], [0.14, 0.17], [0.5, 1.5],   [1.8, 2.2],   2.0,  0.04, ], #3
        [[0.0, 0.0], [0.00, 0.00], [-1.5, -0.5], [1.8, 2.2],   10.0, 0.15, ], #4
        [[0.0, 0.0], [0.00, 0.00], [0.0, 0.0],   [0.9, 1.1],   8.0,  0.10, ], #5
        [[0.0, 0.0], [0.00, 0.00], [0.5, 1.5],   [9.0, 11.0],  11.0, 0.38, ], #6
        [[0.5, 0.6], [0.00, 0.00], [0.0, 0.0],   [32.0, 40.0], 43.0, 0.15, ], #7
        [[0.0, 0.0], [0.00, 0.00], [0.0, 0.0],   [14.0, 18.0], 23.0, 0.34, ], #8
        [[4.0, 4.8], [0.40, 0.48], [-2.5, -1.5], [3.5, 4.5],   20.0, 0.46, ], #9
        ]

obj_labels = ['P', 'N', 'Climate', 'Savings', 'Cost [-]', 'Water']
def follow_up(pred, post):
    if post:
        if not pred:
            return False
    return True

def mutual_exclusive(a, b):
    if a and b:
        return False
    return True

    
def of(x, budget_limit=budget_limit):
    # add constraints
    error_out = (np.nan,)*6
    # follow up action
    if not follow_up(x[3], x[4]):
        return error_out
    
    # Mutually exclusive actions
    if not mutual_exclusive(x[3], x[5]):
        return error_out
    
    if not mutual_exclusive(x[6], x[7]):
        return error_out
    
    if not mutual_exclusive(x[6], x[8]):
        return error_out
 
    # Budget and water constraints
    budget = np.sum([a[-2] for i, a in enumerate(action) if x[i]]) 
    if budget > budget_limit:
        return error_out

    water = 1.0 - np.prod([(1.0 - a[-1]) for i, a in enumerate(action) if x[i]])
    if water < 0.5:
        return error_out
    
    # calculate values
    p = np.sum([[a[0][0], a[0][1]] for i, a in enumerate(action) if x[i]], axis=0)
    n = np.sum([[a[1][0], a[1][1]] for i, a in enumerate(action) if x[i]], axis=0)
    climate = np.sum([[a[2][0], a[2][1]] for i, a in enumerate(action) if x[i]], axis=0)
    savings = np.sum([[a[3][0], a[3][1]] for i, a in enumerate(action) if x[i]], axis=0)
    
    return p, n, climate, savings, budget, water

x = [1,1,1,0,0,1,0,0,1]
of(x)

# Generate all potential portfolios
inp_comb = itertools.product([0, 1], repeat=len(x))

# Get solutions with no NAN values
temp = [[of(i), i] for i in inp_comb if of(i)[-1] is not np.nan]

sols = [list(t[0]) for t in temp]
inp = np.array([list(t[1]) for t in temp])

# turn all of the solutions into a maximisation problem. change the direciton of cost
for i in range(len(sols)):
    sols[i][-2] *= -1  # Minimise the cost

# plot the pareto sets
max_sols = np.array([[np.max(i) for i in soli] for soli in sols])
min_sols = np.array([[np.min(i) for i in soli] for soli in sols])

pf = np.where(utils.pareto_fronts(max_sols[:,:4], minimize=False) == 0)[0]

# for i in range(max_sols.shape[1]):
#     for j in range(i+1, max_sols.shape[1]):
#         for k in range(max_sols.shape[0]):
#             plt.plot([max_sols[k,i], max_sols[k,j]], 
#                      [min_sols[k,i], min_sols[k,j]])
            
#         plt.xlabel(obj_labels[i])
#         plt.ylabel(obj_labels[j])
#         plt.grid()
#         plt.show()
    
for i in range(max_sols.shape[1]):
    for j in range(i+1, max_sols.shape[1]):
        plt.plot(max_sols[:,i], max_sols[:,j], '.')
        plt.plot(max_sols[pf,i], max_sols[pf,j], 'or')            
        plt.xlabel(obj_labels[i])
        plt.ylabel(obj_labels[j])
        plt.grid()
        plt.show()

#%%
def core_index(sols_inp, pf):
    return np.mean(sols_inp[pf,:], axis=0)
core_index(inp, pf)

#%%

# Make scenarios using 


# Make barplot of the pareto optimal solutions
plt.bar(range(len(obj_labels)), max_sols[pf[0],:])
plt.bar(range(len(obj_labels)), min_sols[pf[0],:], color='w')

# 