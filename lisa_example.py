# -*- coding: utf-8 -*-
"""
Created on Wed May  1 08:57:15 2019

@author: jchaconhurtado
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.random import beta, normal, lognormal, logistic, uniform
from scipy.stats import truncnorm
import utils

import generator

# DEfine the generator of the tuncated normal distributions
def tn(mu, sigma, lower=-np.inf, upper=np.inf):
    out = truncnorm((lower - mu) / sigma, 
                    (upper - mu) / sigma, 
                    loc=mu, 
                    scale=sigma)
    return out.rvs()



# objectives
objectives = [
        [0.0, 100.0],  #rehab - max
        [0.0, 100.0],  #adapt - max
        [0.0, 180.0],  #gwhh - min
        [0.0, 2.000],  #econs - min
        [0.0, 365.0],  #vol_dw - min
        [500.0, 3600.0],  #vol_ffw - max
        [0.0, 0.25],  #reliab_dw - min
        [0.0, 365.0],  #aes_dw - min
        [0.0, 365.0],  #faecal_dw - min
        [0.0, 2.0],  #cells_dw - min
        [0.0, 20.0],  #no3_dw - min
        [0.0, 0.02],  #pest - min
        [0.0, 150.0],  #bta_dw - min
        [20.0, 95.0],  #efqm - max
        [0.0, 100.0],  #voice - max
        [0.0, 100.0],  #auton - max
        [0.0, 10.0],  #time - min
        [0.0, 10.0],  #area - min
        [1.0, 6.0],  #collab - max
        [0.01, 5.0],  #costcap - min
        [0.0, 5.0],  #costchange - min
            ]

# always 11 objectives

generator.status_quo()
