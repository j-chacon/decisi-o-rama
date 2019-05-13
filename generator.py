# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:22:05 2019

@author: jchaconhurtado
"""
import numpy as np
from numpy.random import beta, normal, lognormal, uniform
from scipy.stats import truncnorm



# DEfine the generator of the tuncated normal distributions
def tn(mu, sigma, lower=-np.inf, upper=np.inf, size=None):
    out = truncnorm((lower - mu) / sigma, 
                    (upper - mu) / sigma, 
                    loc=mu, 
                    scale=sigma)
    return out.rvs(size=size)

# for the aggregation model
def alpha(n=None): 
    return uniform(0,1, size=n)

def curvature(n=None):
    return uniform(-1, 1, size=n)

# action labels
act_labels = ['A1a','A1b','A2','A3','A4','A5','A6','A7','A8a','A8b','A9']
obj_labels = ['rehab', 'adapt', 'gwhh', 'econs', 'vol_dw', 'vol_hw', 'vol_ffw',
              'reliab_dw', 'reliab_hw', 'reliab_ffw', 'aes_dw', 'aes_hw', 
              'faecal_dw', 'faecal_hw', 'cells_dw', 'cells_hw', 'no3_dw',
              'pest', 'bta_dw', 'efqm', 'voice', 'auton', 'time', 'area',
              'collab', 'cost_cap', 'cost_change',]

_keys = ['rehab', 'adapt', 'gwhh', 'econs', 'vol_dw', 'vol_hw', 'vol_ffw', 
         'reliab_dw', 'reliab_hw', 'reliab_ffw', 'aes_dw', 'aes_hw', 
         'faecal_dw', 'faecal_hw', 'cells_dw', 'cells_hw', 'no3_dw', 
         'pest', 'bta_dw', 'efqm', 'voice', 'auton', 'time', 'area', 'collab', 
         'costcap', 'costchange', 'intergen', 'res_gw_prot', 'water_supply', 
         'soc_accept', 'costs', 'dw_supply', 'hw_supply', 'ffw_supply', 
         'dw_quality', 'hw_quality', 'dw_micro_hyg', 'dw_phys_chem', 
         'hw_micro_hyg', 'hw_phys_chem', 'water supply IS']

_primary_keys = ['rehab', 'adapt', 'gwhh', 'econs', 'vol_dw', 'vol_hw', 
                 'vol_ffw', 'reliab_dw', 'reliab_hw', 'reliab_ffw', 'aes_dw', 
                 'aes_hw', 'faecal_dw', 'faecal_hw', 'cells_dw', 'cells_hw', 
                 'no3_dw', 'pest', 'bta_dw', 'efqm', 'voice', 'auton', 'time', 
                 'area', 'collab', 'costcap', 'costchange']
# class objective():
#     def __init__(self, childs=None, weights=None, label=None, sols )



# always 27 objectives
obj_maximise = [True, True, False, False, False, False, True, False, False, 
                False, False, False, False, False, False, False, False, False, 
                False, True, True, True, False, False, True, False, False]

obj_limits = np.array([
        [0.0, 100.0],  #rehab - max  - 0
        [0.0, 100.0],  #adapt - max  - 1
        [0.0, 180.0],  #gwhh - min  - 2
        [0.0, 2.000],  #econs - min  - 3
        [0.0, 365.0],  #vol_dw - min - 4
        [0.0, 365.0],  #vol_hw - min  - 5
        [500.0, 3600.0],  #vol_ffw - max  - 6
        [0.0, 0.25],  #reliab_dw - min  - 7
        [0.0, 0.25],  #reliab_hw - min  - 8
        [0.0, 0.25],  #reliab_ffw - min - 9
        [0.0, 365.0],  #aes_dw - min  - 10
        [0.0, 365.0],  #aes_hw - min  - 11
        [0.0, 365.0],  #faecal_dw - min  - 12
        [0.0, 365.0],  #faecal_hw - min  - 13
        [0.0, 2.0],  #cells_dw - min - 14
        [0.0, 2.0],  #cells_hw - min  - 15
        [0.0, 20.0],  #no3_dw - min  - 16
        [0.0, 0.02],  #pest - min  - 17
        [0.0, 150.0],  #bta_dw - min  - 18
        [20.0, 95.0],  #efqm - max - 19
        [0.0, 100.0],  #voice - max  - 20
        [0.0, 100.0],  #auton - max  - 21
        [0.0, 10.0],  #time - min  - 22
        [0.0, 10.0],  #area - min  - 23
        [1.0, 6.0],  #collab - max - 24
        [0.01, 5.0],  #costcap - min  - 25
        [0.0, 5.0],  #costchange - min  - 26
            ])

# min, mean, max
_wg_vals = dict(rehab =[0, 0.52, 0.83],	  # rehab
                adapt = [0, 0.38, 0.77],  # adapt
                gwhh = [0.38, 0.73, 1.0],  # gwhh
                econs = [0.0, 0.28,0.63],  # econs
                vol_dw = [0.0, 0.22, 0.36], # vol_dw
                vol_hw = [0.14, 0.28, 0.48], # vol_hw
                vol_ffw = [0.0, 0.34, 0.50], # vol_ffw
                reliab_dw = [0.15, 0.33, 0.48], # reliab_dw
                reliab_hw = [0.26, 0.42, 0.59], # reliab_hw
                reliab_ffw = [0.0, 0.56, 0.83], # reliab_ffw
                aes_dw = [0.07, 0.3, 0.45], # aes_dw
                aes_hw = [0.19, 0.41, 0.83], # aes_hw
                faecal_dw = [0.5, 0.68, 1.0], # faecal_dw  == dw_hygiene
                faecal_hw = [0.5, 0.68, 1.0], # faecal_hw  == hw hygiene
                cells_dw = [0.0, 0.33, 0.50], # cells_dw  = microbial regrowth
                cells_hw = [0.0, 0.32, 0.50], # cells_hw
                no3_dw = [0.0, 0.27, 1.0], # no3_dw  == inorganics
                pest = [0.0, 0.34, 1.0], # pest  = pest_dw
                bta_dw = [0.0, 0.42, 1.0], # bta_dw  = micropollutants
                efqm = [0.0, 0.25, 0.83], # efqm  = operational management
                voice = [0.0, 0.11, 0.29], # voice == codetermination
                auton = [0.0, 0.11, 0.33], # auton
                time = [0.0, 0.1, 0.28], # time
                area = [0.0, 0.09, 0.28], # area
                collab = [0.0, 0.14, 0.33], # collab  = unnecesary disturbance
                costcap = [0.23, 0.54, 1.0], # costcap
                costchange = [0.29, 0.33, 0.38], # costchange
                
                # from this point on are the higher level weights
                # First_level
                intergen = [0.0, 0.19, 0.34],
                res_gw_prot = [0.06, 0.24, 0.48],
                water_supply = [0.23, 0.33, 0.43],
                soc_accept = [0.0, 0.08, 0.23],
                costs = [0.07, 0.18, 0.23],
                
                # second_level
                dw_supply = [0.28, 0.48, 0.83],
                hw_supply = [0.07, 0.29, 0.43],
                ffw_supply = [0.0, 0.24, 0.43],
                
                # Third level
                dw_quality = [0.30, 0.45, 0.83],
                hw_quality = [0.05, 0.31, 0.54],
                
                # fourth level
                dw_micro_hyg = [0.33, 0.44, 0.71],
                dw_phys_chem = [0.05, 0.26, 0.36],
                hw_micro_hyg = [0.07, 0.44, 0.67],
                hw_phys_chem = [0.0, 0.15, 0.36],
            )


def weights(n=None):
    # get weights
    out = {}
    for wg in _wg_vals.keys():
        out[wg] = tn(_wg_vals[wg][1], 
           (_wg_vals[wg][2] - _wg_vals[wg][0])/3.9, 
           0.0, 1.0, n)
    return out

weights(2)

def sq_rehab():
    return status_quo()[0]

def sq_adapt():
    return status_quo()[1]

def status_quo(n=None):
    status_quo = [[ # rehab
        beta(9.0375, 4.0951, size=n),
        beta(9.0375, 4.0951, size=n),
        beta(19.0754,8.9788, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        beta(19.0754, 8.9788, size=n),
        uniform(0,0, size=n),
        normal(0.0438, 0.0162, size=n),
        normal(0.0438, 0.0162, size=n),
        uniform(0,0, size=n),
              ],[ # adapt
        normal(35.0, 7.65, size=n),
        normal(40.0, 10.2, size=n),
        normal(20.0, 10.2, size=n),
        normal(85.0, 7.65, size=n),
        normal(62.5, 6.38, size=n),
        normal(62.5, 6.38, size=n),
        normal(55.0, 7.65, size=n),
        normal(65.0, 7.65, size=n),
        normal(35.0, 7.65, size=n),
        normal(35.0, 7.65, size=n),
        normal(30.0, 10.2, size=n),
          ],[ # gwhh
        normal(6.45, 1.08, size=n),
        normal(6.45, 1.08, size=n),
        normal(6.45, 1.08, size=n),
        normal(5.32, 0.89, size=n),
        normal(6.45, 1.08, size=n),
        normal(11.0, 1.84, size=n),
        normal(8.49, 1.42, size=n),
        normal(6.45, 1.08, size=n),
        normal(6.45, 1.08, size=n),
        normal(6.45, 1.08, size=n),
        normal(6.45, 1.08, size=n),
            ], [ # econs
        normal(0.713, 0.1783, size=n),
        normal(0.713, 0.1783, size=n),
        normal(0.713, 0.1783, size=n),
        normal(0.0777, 0.0194, size=n),
        normal(0.4, 0.1, size=n),
        normal(0.3649, 0.0912, size=n),
        normal(0.55, 0.1375, size=n),
        normal(0.185, 0.0462, size=n),
        normal(0.67, 0.1675, size=n),
        normal(0.67, 0.1675, size=n),
        normal(0.67, 0.1675, size=n),
            ], [# vol_dw
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
            ], [  #vol_hw
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
            ], [ # vol_ffw
        normal(1766.968, 442.0, size=n),
        normal(1766.968, 442.0, size=n),
        normal(1310.211, 328.0, size=n),
        normal(1726.288, 432.0, size=n),
        normal(1766.968, 442.0, size=n),
        normal(1838.676, 460.0, size=n),
        normal(1310.211, 328.0, size=n),
        normal(1838.676, 460.0, size=n),
        normal(1766.968, 442.0, size=n),
        normal(1766.968, 442.0, size=n),
        normal(1310.211, 328.0, size=n),
            ], [ #reliab_dw
        lognormal(-5.2162, 0.2991, size=n),
        lognormal(-5.2162, 0.2991, size=n),
        lognormal(-5.1793, 0.3056, size=n),
        uniform(0.98,1.0, size=n),
        normal(0.0827, 0.0161, size=n),
        normal(0.175, 0.0375, size=n),
        lognormal(-5.1793, 0.3056, size=n),
        normal(0.065, 0.0175, size=n),
        lognormal(-4.2198, 0.3378, size=n),
        lognormal(-4.2198, 0.3378, size=n),
        lognormal(-4.0617, 0.3748, size=n),
            ], [ # relihab_hw
        lognormal(-5.2162, 0.2991, size=n),
        lognormal(-5.2162, 0.2991, size=n),
        lognormal(-5.1793, 0.3056, size=n),
        normal(0.65, 0.0175, size=n),
        lognormal(-4.0617, 0.3748, size=n),
        normal(0.175, 0.0375, size=n),
        lognormal(-5.1793, 0.3056, size=n),
        normal(0.065, 0.0175, size=n),
        lognormal(-4.2198, 0.3378, size=n),
        lognormal(-4.2198, 0.3378, size=n),
        lognormal(-4.0617, 0.3748, size=n),
            ],[ #reliab_ffw
        lognormal(-5.2162, 0.2991, size=n),
        lognormal(-5.2162, 0.2991, size=n),
        lognormal(-5.1793, 0.3056, size=n),
        normal(0.65, 0.0175, size=n),
        lognormal(-4.0617, 0.3748, size=n),
        normal(0.175, 0.0375, size=n),
        lognormal(-5.1793, 0.3056, size=n),
        normal(0.065, 0.0175, size=n),
        lognormal(-4.2198, 0.3378, size=n),
        lognormal(-4.2198, 0.3378, size=n),
        lognormal(-4.0617, 0.3748, size=n),
            ], [ # aes_dw
        normal(5.0, 2.55, size=n),
        normal(5.0, 2.55, size=n),
        normal(5.0, 2.55, size=n),
        normal(1.0, 0.51, size=n),
        normal(1.0, 0.51, size=n),
        normal(20.0, 5.1, size=n),
        normal(5.0, 2.55, size=n),
        normal(27.5, 11.48, size=n),
        normal(5.0, 2.55, size=n),
        normal(5.0, 2.55, size=n),
        normal(10.0, 5.1, size=n),
            ],[  # aes_hw
        normal(5.0, 2.55, size=n),
        normal(5.0, 2.55, size=n),
        normal(5.0, 2.55, size=n),
        normal(55.0, 22.96, size=n),
        normal(75.0, 12.76, size=n),
        normal(20.0, 5.1, size=n),
        normal(10.0, 5.1, size=n),
        normal(27.5, 11.48, size=n),
        normal(5.0, 2.55, size=n),
        normal(5.0, 2.55, size=n),
        normal(10.0, 5.1, size=n),            
            ], [  # faecal_dw
        normal(2.5, 1.28, size=n),
        normal(2.5, 1.28, size=n),
        normal(2.5, 1.28, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        normal(1.0, 0.51, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        normal(2.5, 1.28, size=n),
        normal(2.5, 1.28, size=n),
        normal(5.0, 2.55, size=n),
            ], [ # faecal_hw
        normal(2.5, 1.28, size=n),
        normal(2.5, 1.28, size=n),
        normal(2.5, 1.28, size=n),
        uniform(0,0, size=n),
        normal(20.0, 5.1, size=n),
        normal(1.0, 0.51, size=n),
        normal(5.0, 2.55, size=n),
        uniform(0,0, size=n),
        normal(2.5, 1.28, size=n),
        normal(2.5, 1.28, size=n),
        normal(5, 2.55, size=n),    
            ], [  # cells_dw
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        normal(0.15, 0.08, size=n),
        normal(-0.5, 0.26, size=n),
        normal(-1.5, 0.26, size=n),
        normal(0.14, 0.07, size=n),
        normal(0.34, 0.07, size=n),
        normal(0.1, 0.05, size=n),
        normal(0.1, 0.05, size=n),
        normal(0.15, 0.08, size=n),
            ], [ # cells_hw
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        normal(0.1, 0.05, size=n),
        normal(0.39, 0.05, size=n),
        normal(0.35, 0.18, size=n),
        normal(-1.5, 0.26, size=n),
        normal(0.24, 0.03, size=n),
        normal(0.34, 0.07, size=n),
        normal(0.1, 0.05, size=n),
        normal(0.1, 0.05, size=n),
        normal(0.15, 0.08, size=n),
            ], [ #no3
        uniform(0.0, 20.0, size=n),
        uniform(0.0, 20.0, size=n),
        uniform(0.0, 20.0, size=n),
        uniform(0.0, 20.0, size=n),
        uniform(0.0, 20.0, size=n),
        uniform(0.0, 20.0, size=n),
        uniform(0.0, 20.0, size=n),
        uniform(0.0, 20.0, size=n),
        uniform(0.0, 20.0, size=n),
        uniform(0.0, 20.0, size=n),
        uniform(0.0, 20.0, size=n),
            ], [  # pest
        uniform(0.0, 0.02, size=n),
        uniform(0.0, 0.02, size=n),
        uniform(0.0, 0.02, size=n),
        uniform(0.0, 0.02, size=n),
        uniform(0.0, 0.02, size=n),
        uniform(0.0, 0.02, size=n),
        uniform(0.0, 0.02, size=n),
        uniform(0.0, 0.02, size=n),
        uniform(0.0, 0.02, size=n),
        uniform(0.0, 0.02, size=n),
        uniform(0.0, 0.02, size=n),
            ], [  # bta_dw
        uniform(0.0, 150.0, size=n),
        uniform(0.0, 150.0, size=n),
        uniform(0.0, 150.0, size=n),
        uniform(0.0, 150.0, size=n),
        uniform(0.0, 150.0, size=n),
        uniform(0.0, 150.0, size=n),
        uniform(0.0, 150.0, size=n),
        uniform(0.0, 150.0, size=n),
        uniform(0.0, 150.0, size=n),
        uniform(0.0, 150.0, size=n),
        uniform(0.0, 150.0, size=n),
            ], [ # eqfm
        normal(68.0, 6.63, size=n),
        normal(72.0, 6.63, size=n),
        normal(69.0, 4.59, size=n),
        normal(37.0, 5.61, size=n),
        normal(39.0, 7.65, size=n),
        normal(33.0, 5.61, size=n),
        normal(65.0, 2.55, size=n),
        normal(62.0, 5.1, size=n),
        normal(63.0, 2.55, size=n),
        normal(63.0, 2.55, size=n),
        normal(46.0, 8.16, size=n),
            ],[  # voice
        normal(20.0, 10.2, size=n),
        normal(40.0, 10.2, size=n),
        normal(50.0, 4.51, size=n),
        normal(80.0, 10.2, size=n),
        normal(70.0, 15.31, size=n),
        normal(80.0, 10.2, size=n),
        normal(60.0, 10.2, size=n),
        normal(75.0, 12.76, size=n),
        normal(70.0, 10.2, size=n),
        normal(70.0, 10.2, size=n),
        normal(80.0, 10.2, size=n),
            ], [ #auton
        uniform(55.1981, 55.1981, size=n),
        uniform(55.2, 55.2, size=n),
        uniform(55.2, 55.2, size=n),
        uniform(80.32, 80.32, size=n),
        uniform(55.46, 55.46, size=n),
        uniform(100.0, 100.0, size=n),
        uniform(90.0, 90.0, size=n),
        uniform(89.33, 89.33, size=n),
        uniform(55.46, 55.46, size=n),
        uniform(55.4571, 55.4571, size=n),
        uniform(55.46, 55.46, size=n),
            ], [  # time
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0.36, 0.36, size=n),
        uniform(1.69, 1.69, size=n),
        uniform(5.0, 5.0, size=n),
        uniform(8.04, 8.04, size=n),
        uniform(0.36, 0.36, size=n),
        uniform(1.69, 1.69, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
        uniform(0,0, size=n),
            ], [# area
        uniform(0.0, 0.0, size=n),
        uniform(0.0, 0.0, size=n),
        uniform(0.0, 0.0, size=n),
        uniform(7.35, 7.35, size=n),
        uniform(0.25, 0.25, size=n),
        uniform(5.63, 5.63, size=n),
        uniform(6.78, 6.78, size=n),
        uniform(7.09, 7.09, size=n),
        uniform(0.0, 0.0, size=n),
        uniform(0.0, 0.0, size=n),
        uniform(0.0, 0.0, size=n),
            ], [ # collab
        uniform(6.0, 6.0, size=n),
        uniform(6.0, 6.0, size=n),
        uniform(6.0, 6.0, size=n),
        uniform(1.0, 1.0, size=n),
        uniform(1.0, 1.0, size=n),
        uniform(2.0, 2.0, size=n),
        uniform(6.0, 6.0, size=n),
        uniform(6.0, 6.0, size=n),
        uniform(2.0, 2.0, size=n),
        uniform(2.0, 2.0, size=n),
        uniform(1.0, 1.0, size=n),
            ], [ # costcap
        lognormal(-5.1776, 0.1232, size=n),
        lognormal(-5.1776, 0.1232, size=n),
        tn(0.0039, 0.0006, 0.002, 0.007, size=n),
        lognormal(-4.2529, 0.2835, size=n),
        lognormal(-5.6495, 0.1676, size=n),
        lognormal(-5.0688, 0.3677, size=n),
        tn(0.0039, 0.0006, 0.002, 0.006, size=n),
        lognormal(-4.7923, 0.2947, size=n),
        lognormal(-5.5707, 0.1603, size=n),
        lognormal(-5.5707, 0.1603, size=n),
        beta(25.88, 8599.462, size=n),
            ], [  # costchange
        normal(0.0062, 0.0003, size=n),
        normal(0.0062, 0.0003, size=n),
        normal(0.0043, 0.0002, size=n),
        normal(0.0043, 0.0002, size=n),
        normal(0.0038, 0.0002, size=n),
        normal(0.0074, 0.0004, size=n),
        normal(0.0043, 0.0002, size=n),
        normal(0.0094, 0.0005, size=n),
        normal(0.0042, 0.0002, size=n),
        normal(0.0042, 0.0002, size=n),
        normal(0.0032, 0.0001, size=n),
            ]]
    return np.array(status_quo)