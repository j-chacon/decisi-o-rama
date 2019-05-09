# -*- coding: utf-8 -*-
"""
Created on Wed May  1 08:57:15 2019

@author: jchaconhurtado
"""


import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
# from numpy.random import beta, normal, lognormal, uniform
#from scipy.stats import rankdata
import itertools
import utils
#import seaborn as sns
import generator
#from sklearn.preprocessing import MinMaxScaler
#


n = 1000
sq_sols = np.array(generator.status_quo(n=n))
#%%
# primary solutions
def sol_gen(x):    
    out = np.zeros([sq_sols.shape[0], n])
    for i in range(sq_sols.shape[2]):
        out[:, i] = np.dot(sq_sols[:,:,i], np.array(x))
    
    for i in range(out.shape[0]):
        out[i, :] = np.clip(out[i, :], 
           generator.obj_limits[i,0], 
           generator.obj_limits[i,1])
        # print(generator.obj_limits[i,0])
    return out


x =  np.ones(11)
y = sol_gen(x)

# Generate all the solutions
inp_comb = itertools.product([0, 1], repeat=len(x))
inps = np.array([i for i in inp_comb])
sols = np.array([sol_gen(inp) for inp in inps])

sols_dict = {}
for i, key in enumerate(generator._primary_keys):
    sols_dict[key] = sols[:, i, :]

# weights = generator.weights(n)

#%%
sorted_solutions = np.copy(sols)
# Turn the problem into a minimisation problem
for i, max_true in enumerate(generator.obj_maximise):
    if max_true:
        sorted_solutions[:, i, :] *= -1


#%%        
scaled_sols = np.zeros_like(sorted_solutions)
# range_scale the sorted solutions
for i in range(sols.shape[1]):
    if generator.obj_maximise[i]:  # direction
        
        scaled_sols[:,i,:] = ((sols[:,i,:] - generator.obj_limits[i, 0]) / 
                              (generator.obj_limits[i, 1] - generator.obj_limits[i, 0]))
    else:
        scaled_sols[:,i,:] = 1.0 -1.0*((sols[:,i,:] - generator.obj_limits[i, 0]) / 
                              (generator.obj_limits[i, 1] - generator.obj_limits[i, 0]))

#%%        
#turn into utility
curvature = generator.curvature(n=sols.shape)
util_vals = utils.util_exponential(scaled_sols, curvature)

util_vals_dict = {}
for i, key in enumerate(generator._primary_keys):
    util_vals_dict[key] = sols[:, i, :]

#%%
#aggregate the first two utilities (in the shape of the solutions)
# agg_sols = [0,1] # rehab


def fun_agg(candidates, node_out):
    ''''''
    # check if candidates exist
    if not set(candidates).issubset(generator._keys):
        raise ValueError('One of the candidate nodes does not exist')
    
    # Check for nodes in the node_out
    if node_out not in generator._keys:
        raise ValueError('node_out is not does not exist')
        
    _w = np.array([w[c] for c in candidates])
    _agg_members = np.array([util_vals_dict[c] for c in candidates])
    alpha = generator.alpha(n)
    # w = generator.weights(n)[agg_sols]
    # agg_members = util_vals[:,agg_sols,:]
    
    add_model = np.zeros([sols.shape[0], sols.shape[2]])
    cd_model = np.zeros([sols.shape[0], sols.shape[2]])
    # joint_model = np.zeros([sols.shape[0], sols.shape[2]])
    
    
    add_model = utils.pref_additive(_agg_members, _w, w_norm=True)
    cd_model = utils.pref_cobb_douglas(_agg_members, _w, w_norm=True)   
    util_vals_dict[node_out] = alpha * add_model + (1.0 - alpha)*cd_model
    if np.any(np.isnan(util_vals_dict[node_out])):
        print('we got a nan')
    return 

w = generator.weights(n)

_keys = ['rehab', 'adapt', 'gwhh', 'econs', 'vol_dw', 'vol_hw', 'vol_ffw', 
         'reliab_dw', 'reliab_hw', 'reliab_ffw', 'aes_dw', 'aes_hw', 
         'faecal_dw', 'faecal_hw', 'cells_dw', 'cells_hw', 'no3_dw', 
         'pest', 'bta_dw', 'efqm', 'voice', 'auton', 'time', 'area', 'collab', 
         'costcap', 'costchange', 'intergen', 'res_gw_prot', 'water_supply', 
         'soc_accept', 'costs', 'dw_supply', 'hw_supply', 'ffw_supply', 
         'dw_quality', 'hw_quality', 'dw_micro_hyg', 'dw_phys_chem', 
         'hw_micro_hyg', 'hw_phys_chem', 'water supply IS']

fun_agg(['reliab_ffw', 'vol_ffw'], 'ffw_supply')

# w1
fun_agg(['rehab', 'adapt'], 'intergen')

# w2
fun_agg(['gwhh', 'econs'], 'res_gw_prot')

# w3
fun_agg(['faecal_dw', 'cells_dw'], 'dw_micro_hyg')
fun_agg(['no3_dw', 'pest', 'bta_dw'], 'dw_phys_chem')
fun_agg(['aes_dw', 'dw_micro_hyg', 'dw_phys_chem'], 'dw_quality')
fun_agg(['vol_dw','reliab_dw','dw_quality'], 'dw_supply')

fun_agg(['faecal_hw', 'cells_hw'], 'hw_micro_hyg')
fun_agg(['no3_dw', 'pest', 'bta_dw'], 'hw_phys_chem')
fun_agg(['aes_hw', 'hw_micro_hyg', 'hw_phys_chem'], 'hw_quality')
fun_agg(['vol_hw', 'reliab_hw', 'hw_quality'], 'hw_supply')

fun_agg(['reliab_ffw', 'vol_ffw'], 'ffw_supply')

fun_agg(['dw_supply', 'hw_supply', 'ffw_supply'], 'water_supply')

# w4
fun_agg(['efqm', 'voice', 'auton', 'time', 'area', 'collab'], 'soc_accept')

# w5
fun_agg(['costcap', 'costchange'], 'costs')

# complete aggregation
fun_agg(['intergen','res_gw_prot','water_supply','soc_accept','costs'], 'water supply IS')


res = util_vals_dict['water supply IS']

#%%rank the solutions 
rank = np.zeros_like(res)
for i in range(n):
    rank[:, i] = np.argsort(res[:, i])[::-1]  # reversed list of sorted indexes

plt.plot(rank)

#%% Sort by median
med_rank = np.median(res, axis=1)  # Maximise median utility
iqr = np.quantile(res, 0.75, axis=1) - np.quantile(res, 0.25, axis=1)  # Minimise interquantile

# make the vectors a decision problem - maximisation
dp = np.vstack([med_rank, -iqr]).T

# get dominating solutions
pf = utils.pareto_front_i(dp, minimize=False)


#%%
# Visualise the alternatives
# calculate core index
ci = utils.core_index(inps, pf)
plt.bar(range(len(ci)), ci)
plt.xticks(range(len(ci)), generator.act_labels)
plt.ylabel('Core Index [-]')
plt.grid()

#%%
med_rank_idx = np.argsort(med_rank)[::-1]

# build ranked solutions
ranked_sols = inps[med_rank_idx]

#plt.plot(ranked_sols)
#%%

# #%%
# def fun_agg(agg_members, w):
#     alpha = generator.alpha(n)
#     # w = generator.weights(n)[agg_sols]
#     # agg_members = util_vals[:,agg_sols,:]
    
#     add_model = np.zeros([sols.shape[0], sols.shape[2]])
#     cd_model = np.zeros([sols.shape[0], sols.shape[2]])
#     # joint_model = np.zeros([sols.shape[0], sols.shape[2]])
    
#     for i in range(sols.shape[2]):
#         add_model[:,i] = utils.pref_additive(agg_members[:,:,i], w[:, i], w_norm=True)
#         cd_model[:,i] = utils.pref_cobb_douglas(agg_members[:,:,i], w[:, i], w_norm=True)
    
#     out = alpha * add_model + (1.0 - alpha)*cd_model
#     return out

# w = generator.weights(n)

# # w1
# _w = np.array([w['rehab'], w['adapt']])
# gg = np.array([util_vals_dict['rehab'], util_vals_dict['adapt']])
# intergen_equ = fun_agg(gg, _w)

# # w2
# _w = np.array([w['gwhh'], w['econs']])
# res_gw_prot = fun_agg(util_vals[:, [2, 3], :], _w)

# # w3
# _w = np.array([w['gwhh'], w['econs']])
# dw_micro_hyg = fun_agg(util_vals[:, [12, 14], :], )


# dw_phys_chem = fun_agg(util_vals[:, [16, 17, 18], :], w[[16, 17, 18]])


# hw_micro_hyg = fun_agg(util_vals[:, [13, 15], :], w[[13, 15]])


# hw_phys_chem = fun_agg(util_vals[:, [16, 17, 18], :], w[[16, 17, 18]])


# ffw_supply = fun_agg(util_vals[:, [9, 6], :], w[[9, 6]])


# social_accp = fun_agg(util_vals[:, [19,20,21,22,23,24], :], w[[19,20,21,22,23,24]])


# costs = fun_agg(util_vals[:, [25, 26], :], w[[25, 26]])

# w_dw_supply = 

#%%



#%%rank the solutions by the mean
# rank = np.argsort(np.average(joint_model, axis=1))[::-1]  # reversed list of sorted indexes




#%%
# analyse the results for the mean
# # Get the pareto fronts in the mean values
# mean_pf = np.mean(sorted_solutions, axis=2)
# mean_pf0 = utils.pareto_front_i(mean_pf, minimize=True, i=0)
# core_idx = utils.core_index(inps, mean_pf0)
        
# #%%
# plt.figure()
# plt.bar(range(len(core_idx)), core_idx)
# plt.xticks(range(len(core_idx)), generator.act_labels)
# plt.ylabel('Core index')
# plt.grid()
# plt.show()


# #%% make pairplots

# # get vector with domination
# dominance_vec = np.array([0, ]*sorted_solutions.shape[0])
# dominance_vec[mean_pf0] = 1

# mean_sols = np.mean(sols, axis=2)
# df_sols = pd.DataFrame(mean_sols)
# df_sols.columns = generator.obj_labels

# df_sols['dominance'] = dominance_vec

# sns.pairplot(df_sols, hue='dominance', diag_kind='hist')
