# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:28:51 2019

@author: jchaconhurtado
"""
import utils
import numpy as np
#%%

#def test_cobb_douglas():
val = np.array([3600.0, 00.0])
weights = np.array([0.5, 0.5])
utils.pref_cobb_douglas(val, weights)

#%%

utils.pref_cobb_douglas(vals[1,:,:], w[1,:], w_norm=True)

ww = w[1,:]/np.sum(w[1,:])
vv = vals[1,:,:]
gg = utils.pref_cobb_douglas(vv, ww)

vv**ww

#%%
#def cob(sols, w, w_norm=True):

sols = vals
w_norm=True

'''model preferences using cobb-douglas method. takes two vectors'''
# Test this function
if np.sum(w) != 1.0 and w_norm is False:
    raise ValueError('weights have to be normalised to 1')
elif w_norm:
    w = w / np.sum(w, axis=0)

out = np.zeros([sols.shape[1], sols.shape[2]])
for i in range(sols.shape[1]):
    out[i,:] = np.prod(sols[:,i,:]**w, axis=0)



#
#if sols.ndim == 1:
#    out = np.prod(sols)
#elif sols.ndim == 2:
#    out = np.prod(sols, axis=1)
#    
##    return out
#
