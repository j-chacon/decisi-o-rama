# -*- coding: utf-8 -*-
""" Ranker Module

This module contains functions to rank the results of utilities. This can be
used as a part of the multi-criterai decision analysis, or over the results of
the hierarchical aggregation.

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


def iqr(sols, lq=0.25, uq=0.75):
    '''Calculate the interquantile range

    The interquantile range (iqr) is the distance between a lower and upper
    quantile. Larger iqr denote larger spread of the PDF of the vector

    Parameters
    ----------
    sols : ndarray [p, n]
        2D array containing the utility values for all of the portfolios [p],
        and all of the random samples [n]
    lq : float
        Value containing the lower end of the iqr. Cannot be smaller than 0
    uq : float
        Value containing the upper end of the iqr. Cannot be larger than 1

    Returns
    -------
    iqr : ndarray [p]
        1D array containing all of the utility iqr value for each portfolio
    '''
    if uq < lq:
        _msj = 'Upper quanitle has to be larger than the lower quantile'
        raise ValueError(_msj)
    return np.quantile(sols, uq, axis=1) - np.quantile(sols, lq, axis=1)


def mean(sols):
    '''Calculate the mean of the utilities

    This functions calculate the mean of the distributions. It is simply a
    np.average over the first axis.

    Parameters
    ----------
    sols : ndarray [p, n]
        2D array containing the utility values for all of the portfolios [p],
        and all of the random samples [n]

    Returns
    -------
    mean : ndarray [p]
        1D array containing all of the mean utility value for each portfolio
    '''
    return np.average(sols, axis=1)


def std(sols):
    '''Calculate the standard deviation of the utilities

    This functions calculate the standard deviation of the distributions. It
    is simply a np.std over the first axis

    Parameters
    ----------
    sols : ndarray [p, n]
        2D array containing the utility values for all of the portfolios [p],
        and all of the random samples [n]

    Returns
    -------
    std : ndarray [p]
        1D array containing all of the mean utility value for each portfolio
    '''
    return np.std(sols, axis=1)


def cov(sols):
    '''Calculate the coefficient of variation of the utilities

    This functions calculate the coefficient of variation of the
    distributions.

    Parameters
    ----------
    sols : ndarray [p, n]
        2D array containing the utility values for all of the portfolios [p],
        and all of the random samples [n]

    Returns
    -------
    cov : ndarray [p]
        1D array containing all of the coefficient of variation utility
        value for each portfolio
    '''

    return np.std(sols, axis=1)/np.average(sols, axis=1)
