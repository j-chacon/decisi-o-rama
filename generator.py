# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:22:05 2019

@author: jchaconhurtado
"""
import numpy as np
from numpy.random import beta, normal, lognormal, uniform
from scipy.stats import truncnorm


obj_limits = np.array([
        [0.0, 100.0],  #rehab - max
        [0.0, 100.0],  #adapt - max
        [0.0, 180.0],  #gwhh - min
        [0.0, 2.000],  #econs - min
        [0.0, 365.0],  #vol_dw - min - 5
        [0.0, 365.0],  #vol_hw - min
        [500.0, 3600.0],  #vol_ffw - max
        [0.0, 0.25],  #reliab_dw - min
        [0.0, 0.25],  #reliab_hw - min
        [0.0, 0.25],  #reliab_ffw - min - 10
        [0.0, 365.0],  #aes_dw - min
        [0.0, 365.0],  #aes_hw - min
        [0.0, 365.0],  #faecal_dw - min
        [0.0, 365.0],  #faecal_hw - min
        [0.0, 2.0],  #cells_dw - min - 15
        [0.0, 2.0],  #cells_hw - min
        [0.0, 20.0],  #no3_dw - min
        [0.0, 0.02],  #pest - min
        [0.0, 150.0],  #bta_dw - min
        [20.0, 95.0],  #efqm - max - 20
        [0.0, 100.0],  #voice - max
        [0.0, 100.0],  #auton - max
        [0.0, 10.0],  #time - min
        [0.0, 10.0],  #area - min
        [1.0, 6.0],  #collab - max - 25
        [0.01, 5.0],  #costcap - min
        [0.0, 5.0],  #costchange - min
            ])

# DEfine the generator of the tuncated normal distributions
def tn(mu, sigma, lower=-np.inf, upper=np.inf, size=None):
    out = truncnorm((lower - mu) / sigma, 
                    (upper - mu) / sigma, 
                    loc=mu, 
                    scale=sigma)
    return out.rvs(size=size)

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
    return status_quo