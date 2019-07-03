# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:52:06 2019

@author: jchaconhurtado

Implementation of Lahtinen2017 (RPM)

"""
import sys
sys.path.append('..')

import numpy as np
#from decisiorama.utils import random_instance as ri
from decisiorama import pda
import itertools
# Implementation of the problem

n = 200
#ru = ri.Uniform
#nodes = ['p', 'n2', 'climate', 'savings', 'cost', 'water', 'overall']

#%%
#sols = dict(p = [
#            ru(0.9, 1.1).get,
#            ru(1.1, 1.3).get,
#            ru(1.3, 1.7).get,
#            ru(0.0, 0.0).get,
#            ru(0.0, 0.0).get,
#            ru(0.0, 0.0).get,
#            ru(0.5, 0.6).get,
#            ru(0.0, 0.0).get,
#            ru(4.0, 4.8).get,
#            ], n2 = [
#            ru(0.09, 0.11).get,
#            ru(0.09, 0.11).get,
#            ru(0.14, 0.17).get,
#            ru(0.00, 0.00).get,
#            ru(0.00, 0.00).get,
#            ru(0.00, 0.00).get,
#            ru(0.00, 0.00).get,
#            ru(0.00, 0.00).get,
#            ru(0.40, 0.48).get,
#            ], climate = [
#            ru(0.0, 0.0).get,
#            ru(0.0, 0.0).get,
#            ru(0.5, 1.5).get,
#            ru(-1.5, -0.5).get,
#            ru(0.0, 0.0).get,
#            ru(0.5, 1.5).get,
#            ru(0.0, 0.0).get,
#            ru(0.0, 0.0).get,
#            ru(-2.5, -1.5).get,
#            ], savings = [
#            ru(1.8, 2.2).get,
#            ru(1.8, 2.2).get,
#            ru(1.8, 2.2).get,
#            ru(1.8, 2.2).get,
#            ru(0.9, 1.1).get,
#            ru(9.0, 11.0).get,
#            ru(32.0, 40.0).get,
#            ru(14.0, 18.0).get,
#            ru(3.5, 4.5).get,
#            ], overall = None
#            )
#

sols = dict(p=np.array([
            [0.9, 1.1],
            [1.1, 1.3],
            [1.3, 1.7],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.5, 0.6],
            [0.0, 0.0],
            [4.0, 4.8],
            ]), n2=np.array([
            [0.09, 0.11],
            [0.09, 0.11],
            [0.14, 0.17],
            [0.00, 0.00],
            [0.00, 0.00],
            [0.00, 0.00],
            [0.00, 0.00],
            [0.00, 0.00],
            [0.40, 0.48],
            ]), climate = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.5, 1.5],
            [-1.5, -0.5],
            [0.0, 0.0],
            [0.5, 1.5],
            [0.0, 0.0],
            [0.0, 0.0],
            [-2.5, -1.5],
            ]), savings = np.array([
            [1.8, 2.2],
            [1.8, 2.2],
            [1.8, 2.2],
            [1.8, 2.2],
            [0.9, 1.1],
            [9.0, 11.0],
            [32.0, 40.0],
            [14.0, 18.0],
            [3.5, 4.5],
            ]), overall = None
            )

costs = [1.0, 1.0, 2.0, 10.0, 8.0, 11.0, 43.0, 23.0, 20.0]
water = [0.03, 0.07, 0.04, 0.015, 0.10, 0.38, 0.15, 0.34, 0.46]
budget_limit = 45.0
#%%
## Create objectives

p = pda.Objective(
        name = 'p',
        w = 0.25,
        alternatives = sols['p'], 
        obj_min = 0.0, 
        obj_max = 4.8, 
        n = n, 
        utility_func = pda.utility.exponential, 
        utility_pars = [0.0, ], 
        aggregation_func = pda.aggregate.additive, 
        aggregation_pars = None,
        maximise = False)

n2 = pda.Objective(
        name = 'n2',
        w = 0.25,
        alternatives = sols['n2'], 
        obj_min = 0.0, 
        obj_max = 0.48, 
        n = n, 
        utility_func = pda.utility.exponential, 
        utility_pars = [0.0, ], 
        aggregation_func = pda.aggregate.additive, 
        aggregation_pars = None,
        maximise = False)

climate = pda.Objective(
        name = 'climate',
        w = 0.25,
        alternatives = sols['climate'], 
        obj_min = -2.5, 
        obj_max = 1.5, 
        n = n, 
        utility_func = pda.utility.exponential, 
        utility_pars = [0.0, ], 
        aggregation_func = pda.aggregate.additive, 
        aggregation_pars = None,
        maximise = False)

savings = pda.Objective(
        name = 'savings',
        w = 0.25,
        alternatives = sols['savings'], 
        obj_min = 0.9, 
        obj_max = 40.0, 
        n = n, 
        utility_func = pda.utility.exponential, 
        utility_pars = [0.0, ], 
        aggregation_func = pda.aggregate.additive, 
        aggregation_pars = None,
        maximise = False)

overall = pda.Objective(
        name = 'overall',
        w = 0.25,
        alternatives = sols['climate'], 
        obj_min = 0.9, 
        obj_max = 40.0, 
        n = n, 
        utility_func = pda.utility.exponential, 
        utility_pars = 0.0, 
        aggregation_func = pda.aggregate.additive, 
        aggregation_pars = None,
        maximise = False)

overall.add_children(p)
overall.add_children(n2)
overall.add_children(climate) 
overall.add_children(savings)

prob = dict(p=p, n2=n2, climate=climate, savings=savings, 
            overall=overall)

x = [1,1,1,1,1,1,1,1,1]
overall.get_value(x)

# get all the results
inp_comb = list(itertools.product([0, 1], repeat=len(x)))

# Get list of plausible results
def filter_inps(inps):
    out = []
    def follow_up(pred, post):
        if post:
            if not pred:
                return False
        return True
        
    def mutual_exclusive(a, b):
        if a and b:
            return False
        return True

    for x in inps:        
    
        # follow up action
        if not follow_up(x[3], x[4]):
            continue
        
        # Mutually exclusive actions
        if not mutual_exclusive(x[3], x[5]):
            continue
        
        if not mutual_exclusive(x[6], x[7]):
            continue
        
        if not mutual_exclusive(x[6], x[8]):
            continue
     
        # Budget and water constraints
        budget = np.sum([a for i, a in enumerate(costs) if x[i]]) 
        if budget > budget_limit:
            continue
    
        water_target = 1.0 - np.prod([(1.0 - a) for i, a in enumerate(water) if x[i]])
        if water_target < 0.5:
            continue
        
        out.append(x)
    return out
    
inps = np.array(filter_inps(inp_comb))
res = np.array(list(map(overall.get_value, inps)))

ee = pda.Evaluator(inps, res)
ee.add_function(pda.ranker.mean, minimize=False)
ee.add_function(pda.ranker.iqr, minimize=True)
ee.get_pareto_solutions()
