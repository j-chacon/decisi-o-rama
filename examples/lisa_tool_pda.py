# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:58:12 2019

@author: jchaconhurtado
"""

import numpy as np
import pda_fun
import generator
import utility
import aggregate
import random_instance as ri
#import matplotlib.pyplot as plt
import ranker
import itertools
import multiprocessing as mp
from time import time

# 1000 runs about 12 min
# single run 100k 18.5 seg
# in MP 1000 runs in 1.8 min
# in MP 2000 runs in 8.0 min
if __name__ == '__main__':
    n = 100
    n_att = 11
    alternatives = generator.status_quo()
    obj_lim = generator.obj_limits
    wgs = generator.weights()
    alpha = ri.uniform(0,1).get
    multiproc = False
    
    # Create the objectives 
    prob = {}
    for elem in generator._keys:
        prob[elem] = pda_fun.objective(
                name = elem,
                w = wgs[elem],
                alternatives = alternatives[elem], 
                obj_min = obj_lim[elem][0], 
                obj_max = obj_lim[elem][1], 
                n = n, 
                utility_func = utility.exponential, 
                utility_pars = [0.01,], 
                aggregation_func = aggregate.mix_linear_cobb, 
                aggregation_pars = [0.01, ], # alpha
                maximise=generator.obj_maximise[elem])
    
    # Make define map for the children of the nodes
    wsis_map = dict(
            water_supply_IS = ['intergen', 'res_gw_prot', 'water_supply',
                               'soc_accept','costs'],
            intergen = ['rehab', 'adapt'],
            res_gw_prot = ['gwhh', 'econs'],
            water_supply = ['dw_supply', 'hw_supply', 'ffw_supply'],
            soc_accept = ['efqm', 'voice', 'auton', 'time', 'area', 'collab'],
            costs = ['costcap', 'costchange'],
            dw_supply = ['vol_dw', 'reliab_dw', 'dw_quality'],
            hw_supply = ['vol_hw', 'reliab_hw', 'hw_quality'],
            ffw_supply = ['reliab_ffw', 'vol_ffw'],
            dw_quality = ['aes_dw', 'dw_micro_hyg', 'dw_phys_chem'],
            hw_quality = ['aes_hw', 'hw_micro_hyg', 'hw_phys_chem'],
            dw_micro_hyg = ['faecal_dw', 'cells_dw'],
            dw_phys_chem = ['no3_dw', 'pest_dw','bta_dw'],
            hw_micro_hyg = ['faecal_hw', 'cells_hw'],
            hw_phys_chem = ['no3_hw', 'pest_hw', 'bta_hw'],
            )
    
    # add the hierarchy to the problem
    pda_fun.hierarchy_smith(wsis_map, prob)
    
    # Get the inputs (all possible binary combinations)
    inp_comb = itertools.product([0, 1], repeat=n_att)
    inps = np.array([i for i in inp_comb])
    
    # run the decision problem
    f = prob['water_supply_IS'].get_value
    a = time()
    
    if multiproc:
        chunks = (len(inps)//3) + 1
        with mp.Pool(3) as p:
            res = p.map(f, inps, chunks)
    else:
        res = list(map(f, inps))
    
    print(time() - a)
    res = np.array(res)
    
    # From this point we start ranking the solutions
    evaluator = pda_fun.evaluator(inps, res)
    obj_funs = [ranker.mean, ranker.iqr]
    ranked_sols = evaluator.get_ranked_solutions(obj_funs)
    core_index = evaluator.get_core_index(obj_funs)
    
    
    
    
    
    
    
       