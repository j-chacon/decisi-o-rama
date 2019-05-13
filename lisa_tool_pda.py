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

n = 3

# Make the objectives for all the fundamental objectives
rehab = pda_fun.objective(name='rehab', 
                          label='Rehab', 
                          obj_min=0.0, 
                          obj_max=100.0, 
                          w=generator.weights()['rehab'],
                          results=generator.sq_rehab, 
                          n=n, 
                          utility_func=utility.exponential, 
                          utility_pars = [0.01,], 
                          aggregation_func=aggregate.mix_linear_cobb, 
                          aggregation_pars=[0.5,], # alpha
                          maximise=True)

adapt = pda_fun.objective(name='adapt', 
                          label='Adapt', 
                          obj_min=0.0, 
                          obj_max=100.0, 
                          w=generator.weights()['adapt'],
                          results=generator.sq_adapt, 
                          n=n, 
                          utility_func=utility.exponential, 
                          utility_pars = [0.01,], 
                          aggregation_func=aggregate.mix_linear_cobb, 
                          aggregation_pars=[0.5,], # alpha
                          maximise=True)


#rehab.get_value(np.ones(11))
#adapt.get_value(np.ones(11))

intergen = pda_fun.objective(name='intergen', 
                             label='Intergen', 
                             obj_min=0.0, 
                             obj_max=100.0, 
                             w=generator.weights()['intergen'],
                             results=None, 
                             n=n, 
                             utility_func=utility.exponential, 
                             utility_pars = [0.01,], # 
                             aggregation_func=aggregate.mix_linear_cobb, 
                             aggregation_pars=[0.5,], # alpha mix
                             maximise=True)

intergen.add_children(rehab)
intergen.add_children(adapt)

intergen.get_value(np.ones(11))
