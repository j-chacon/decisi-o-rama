# -*- coding: utf-8 -*-
""" Utility Module

This module contains utility functions. Current implementation only has the
exponential utility function.

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


def exponential(v, r, *args, **kwargs):
    '''Calculates the exponential utility function

    Parameters
    ----------
    v : float, ndarray
        Array containing the normalised values
    r : float, ndarray
        Exponent parameter

    returns
    out : ndarray
        Utility values

    Note
    ----
    This is executed as a vectorized function

    '''
    def _exponential(v, r):
        if v > 1.0 or v < 0.0:
            _msj = ('Values passed to the utility function should be ranked '
                    'normalised between 0 and 1')
            RuntimeWarning(_msj)

        if r == 0:
            out = v
        else:
            out = (1.0 - np.exp(-r*v)) / (1.0 - np.exp(-r))
        return out

    _vec_exp = np.vectorize(_exponential)
    return _vec_exp(v, r)
