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
import matplotlib.pyplot as plt

n = 100

sols = generator.status_quo()
obj_lim = generator.obj_limits
sq_rehab = [ # rehab
        ri.beta(9.0375, 4.0951).get,
        ri.beta(9.0375, 4.0951).get,
        ri.beta(19.0754,8.9788).get,
        ri.uniform(0,0).get,
        ri.uniform(0,0).get,
        ri.uniform(0,0).get,
        ri.beta(19.0754, 8.9788).get,
        ri.uniform(0,0).get,
        ri.normal(0.0438, 0.0162).get,
        ri.normal(0.0438, 0.0162).get,
        ri.uniform(0, 0).get,
        ]
wgs = generator.weights()
#rehab_w = ri.truncnormal(0.52, 0.83/3.6, 0).get

sq_adapt = [ # adapt
        ri.normal(35.0, 7.65).get,
        ri.normal(40.0, 10.2).get,
        ri.normal(20.0, 10.2).get,
        ri.normal(85.0, 7.65).get,
        ri.normal(62.5, 6.38).get,
        ri.normal(62.5, 6.38).get,
        ri.normal(55.0, 7.65).get,
        ri.normal(65.0, 7.65).get,
        ri.normal(35.0, 7.65).get,
        ri.normal(35.0, 7.65).get,
        ri.normal(30.0, 10.2).get,
          ]

adapt_w = ri.truncnormal(0.38, 0.77/3.6, 0).get

#%%
# Make the objectives for all the fundamental objectives
xx = 'rehab'
rehab = pda_fun.objective(name=xx, 
                          w=wgs[xx],
                          results=sols[xx], 
                          obj_min=obj_lim[xx][0], 
                          obj_max=obj_lim[xx][1], 
                          n=n, 
                          utility_func=utility.exponential, 
                          utility_pars = [0.01,], 
                          aggregation_func=aggregate.mix_linear_cobb, 
                          aggregation_pars=[0.5,], # alpha
                          maximise=True)
#res = rehab.get_value(np.ones(11))
#plt.hist(res)

#%%
xx = 'adapt'
adapt = pda_fun.objective(name=xx, 
                          w=wgs[xx],
                          results=sols[xx], 
                          obj_min=obj_lim[xx][0], 
                          obj_max=obj_lim[xx][1], 
                          n=n, 
                          utility_func=utility.exponential, 
                          utility_pars = [0.01,], 
                          aggregation_func=aggregate.mix_linear_cobb, 
                          aggregation_pars=[0.5,], # alpha
                          maximise=True)
#adapt.get_value(np.ones(11))
#plt.hist(res)
#%%
xx = 'intergen'
intergen = pda_fun.objective(name=xx, 
                          w=wgs[xx],
                          results=None, 
                          n=n, 
                          utility_func=utility.exponential, 
                          utility_pars = [0.01,], 
                          aggregation_func=aggregate.mix_linear_cobb, 
                          aggregation_pars=[0.5,], # alpha
                          maximise=True)
intergen.add_children(rehab)
intergen.add_children(adapt)

f = intergen.get_value(np.ones(11))
plt.hist(f)



