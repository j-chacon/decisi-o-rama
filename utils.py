# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:42:31 2019

@author: jchaconhurtado
"""

import numpy as np

def pareto_fronts(M, minimize=True):
    '''function to calculate the pareto fronts'''
    if minimize is True:
        i_dominates_j = np.all(M[:,None] <= M, axis=-1) & np.any(M[:,None] < M, axis=-1)
    else:
        i_dominates_j = np.all(M[:,None] >= M, axis=-1) & np.any(M[:,None] > M, axis=-1)
    remaining = np.arange(len(M))
    fronts = np.empty(len(M), int)
    frontier_index = 0
    while remaining.size > 0:
        dominated = np.any(i_dominates_j[remaining[:,None], remaining], axis=0)
        fronts[remaining[~dominated]] = frontier_index

        remaining = remaining[dominated]
        frontier_index += 1
    return fronts