# -*- coding: utf-8 -*-
""" Utils Module

In this module are stored all of the supporting functions which do not fall
into a specific category.

"""

__author__ = "Juan Carlos Chacon-Hurtado"
__credits__ = ["Juan Carlos Chacon-Hurtado", "Lisa Scholten"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Juan Carlos Chacon-Hurtado"
__email__ = "j.chaconhurtado@tudelft.nl"
__status__ = "Development"
__last_update__ = "01-07-2019"

import numpy as np


def pareto_fronts(M, minimize=True):
    '''pareto_fronts

    function to calculate the pareto fronts

    Parameters
    ----------
    M : ndarray
        2D array containing the solution vectors
    minimize : Bool
        Determine if the optimal of the functions is the minimum. In case is a
        maximisation problem, this should be set to False.

    Returns
    -------
    fronts : list
        List containing a list of indexes that represent each solution. The
        "pareto-front" is located in the front 0.
    '''
    if minimize is True:
        i_dominates_j = (np.all(M[:, None] <= M, axis=-1) &
                         np.any(M[:, None] < M, axis=-1))
    else:
        i_dominates_j = (np.all(M[:, None] >= M, axis=-1) &
                         np.any(M[:, None] > M, axis=-1))
    remaining = np.arange(len(M))
    fronts = np.empty(len(M), int)
    frontier_index = 0
    while remaining.size > 0:
        dominated = np.any(i_dominates_j[remaining[:, None], remaining],
                           axis=0)
        fronts[remaining[~dominated]] = frontier_index
        remaining = remaining[dominated]
        frontier_index += 1
    return fronts


def pareto_front_i(M, minimize=True, i=0):
    '''pareto_front_i

    Function to get a specific pareto set. i=0 means optimal

    Parameters
    ----------
    M : ndarray
        2D array containing the solution vectors
    minimize : Bool
        Determine if the optimal of the functions is the minimum. In case is a
        maximisation problem, this should be set to False.
    i : int
        Index that determines the position of the pareto front to retrieve. by
        default is 0, meaning the formal pareto front.

    Returns
    -------
    front : list
        List containing indexes that represent each solution. The
        "pareto-front" is located in the front 0.
    '''
    pfs = pareto_fronts(M, minimize)
    return np.where(pfs == i)[0]


def core_index(sols_inp, pf):
    '''core_index

    calculate the core index. takes solutions and position of
    pareto-solutions

    Parameters
    ----------
    sols_inp : ndarray
        Value of the solutions for each of the activities
    pf : list
        List with the solutions in the pareto front. This list is can be
        obtained througn the `pareto_front_i` function.

    Returns
    -------
    core_index : ndarray
        1D array containing the values of the core index.
    '''
    return np.mean(sols_inp[pf, :], axis=0)
