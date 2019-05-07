# -*- coding: utf-8 -*-
"""
Created on Wed May  1 08:57:15 2019

@author: jchaconhurtado
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from numpy.random import beta, normal, lognormal, uniform
# from scipy.stats import truncnorm
import itertools
import utils
import seaborn as sns
import generator
from sklearn.preprocessing import MinMaxScaler


n = 1000
sq_sols = np.array(generator.status_quo(n=n))
#%%
def sol_gen(x):    
    out = np.zeros([sq_sols.shape[0], n])
    for i in range(sq_sols.shape[2]):
        out[:, i] = np.dot(sq_sols[:,:,i], np.array(x))
    
    for i in range(out.shape[0]):
        out[i, :] = np.clip(out[i, :], 
           generator.obj_limits[i,0], 
           generator.obj_limits[i,1])
        
    return out
    
x =  np.ones(11)
y = sol_gen(x)

# Generate all the solutions
inp_comb = itertools.product([0, 1], repeat=len(x))
inps = np.array([i for i in inp_comb])
sols = np.array([sol_gen(inp) for inp in inps])

#%%
sorted_solutions = sols[:]
# Turn the problem into a minimisation problem
for i, max_true in enumerate(generator.obj_maximise):
    if max_true:
        sorted_solutions[:, i, :] = sorted_solutions[:,i,:]*-1

#%%        
scaled_sols = np.zeros_like(sols)        
# range_scale the sorted solutions
for i in range(sols.shape[1]):
    scaled_sols[:,i,:] = ((sols[:,i,:] - generator.obj_limits[i, 0]) / 
               (generator.obj_limits[i, 1] - generator.obj_limits[i, 0]))

#%%        
#turn into utility
curvature = generator.curvature(n=sols.shape)
util_vals = utils.util_exponential(scaled_sols, curvature)

#%%
# aggregate the utilities ( in the shape of the solutions)
alpha = generator.alpha(n)
w = generator.weights(n)
add_model = np.zeros([sols.shape[0], sols.shape[2]])
cd_model = np.zeros([sols.shape[0], sols.shape[2]])
# joint_model = np.zeros([sols.shape[0], sols.shape[2]])
for i in range(sols.shape[2]):
    add_model[:,i] = utils.pref_additive(util_vals[:,:,i], w[:, i], w_norm=True)
    cd_model[:,i] = utils.pref_cobb_douglas(util_vals[:,:,i], w[:, i], w_norm=True)

joint_model = alpha * add_model + (1.0 - alpha)*cd_model

#%%




#%%
# analyse the results for the mean
# Get the pareto fronts in the mean values
mean_pf = np.mean(sorted_solutions, axis=2)
mean_pf0 = utils.pareto_front_i(mean_pf, minimize=True, i=0)
core_idx = utils.core_index(inps, mean_pf0)
        
#%%
plt.figure()
plt.bar(range(len(core_idx)), core_idx)
plt.xticks(range(len(core_idx)), generator.act_labels)
plt.ylabel('Core index')
plt.grid()
plt.show()


#%% make pairplots

# get vector with domination
dominance_vec = np.array([0, ]*sorted_solutions.shape[0])
dominance_vec[mean_pf0] = 1

mean_sols = np.mean(sols, axis=2)
df_sols = pd.DataFrame(mean_sols)
df_sols.columns = generator.obj_labels

df_sols['dominance'] = dominance_vec

# sns.pairplot(df_sols, hue='dominance', diag_kind='hist')
