# -*- coding: utf-8 -*-
""" Random instance module

This module contain functions to create random numbers on demand given a
predefined distribution.
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
from numpy import random
from scipy.stats import truncnorm


def _isnumeric(x, name):
    '''Check if a given value is numeric'''
    if not isinstance(x, (float, int, np.float, np.float32, np.float64,
                          type(None))):
        msg = '''{0} is not of numeric type. got {1}'''.format(name, type(x))
        raise TypeError(msg)
    return


def _tn(mu, sigma, lower=-np.inf, upper=np.inf, size=None):
    '''help function to create truncated normal distributions'''
    if size is None:
        out = random.normal()
        while lower > out > upper:
            out = random.normal()
    else:
        a = (lower - mu)/sigma
        b = (upper - mu)/sigma
        out = truncnorm.rvs(a, b, mu, sigma, size)
    return out


class RandomNumber():
    '''RandomNumber

    Base class for the generation of random numbers. This will ensure that
    the adequate inheritance of parameters. Do not use this class to create
    random numbers.
    '''
    a = None
    b = None
    n = None
    mu = None
    std = None

    def __init__(self,):
        '''intialisation class is overriden by subclass'''
        pass

    def check_numeric(self):
        '''Check if the values are numeric'''
        tags = [key for key in self.__dict__.keys()]
        for tag in tags:
            _isnumeric(self.__dict__[tag], tag)

    def check_lims(self):
        '''Check that the generated values are within the prescribed limits'''
        if self.a is not None or self.b is not None:
            if self.a > self.b:
                msg = ("The lower limit (a = {0}) cannot be larger than the "
                       "upper limit (b = {1})".format(self.a, self.b))
                raise ValueError(msg)

    def __getstate__(self):
        '''To make the object pickable'''
        state = dict(mu=self.mu,
                     std=self.std,
                     n=self.n,
                     a=self.a,
                     b=self.b,
                     )
        return state

    def __setstate__(self, state):
        '''To make the object pickable'''
        self.mu = state['mu']
        self.std = state['std']
        self.n = state['n']
        self.a = state['a']
        self.b = state['b']
        return


class Normal(RandomNumber):
    '''Class to create Normally distributed random numbers'''

    def __init__(self, mu, std, n=None):
        '''
        Class to create a normally distributed random number

        Parameters
        ----------
        mu : float
            Mean value of the distribution
        std : float
            Standard deviation of the distribution
        n : int, optional
            Number of random numbers to be generated

        '''
        self.mu = mu
        self.std = std
        self.n = n
        self.check_numeric()
        self.check_lims()

    def get(self, n=None):
        '''Method to get the random numbers

        Parameters
        ----------
        n : int, optional
            Number of random numbers to be generated. If passed in the callback
            of this function, it will override the object value.

        Returns
        -------
        out : ndarray
            1D array containing the generated random numbers
        '''
        if n is None and self.n is not None:
            n = self.n
        return random.normal(self.mu, self.std, n)


class Beta(RandomNumber):
    '''Class to create Beta-distributed random numbers'''

    def __init__(self, a, b, n=None):
        '''
        Class to create Beta-distributed random numbers

        Parameters
        ----------
        a : float
            a parameter from the Beta distribution
        b : float
            b parameter from the Beta distribution
        n : int, optional
            Number of random numbers to be generated

        '''
        self.a = a
        self.b = b
        self.n = n
        self.check_numeric()

    def get(self, n=None):
        '''Method to get the random numbers

        Parameters
        ----------
        n : int, optional
            Number of random numbers to be generated. If passed in the callback
            of this function, it will override the object value.

        Returns
        -------
        out : ndarray
            1D array containing the generated random numbers
        '''
        if n is None and self.n is not None:
            n = self.n
        return random.beta(self.a, self.b, n)


class Uniform(RandomNumber):
    '''Class to create Unifomly distributed random numbers'''
    def __init__(self, a, b, n=None):
        '''
        Class to create Beta-distributed random numbers

        Parameters
        ----------
        a : float
            Lower limit of the distribution
        b : float
            Upper limit of the distribution
        n : int, optional
            Number of random numbers to be generated

        '''
        self.a = a
        self.b = b
        self.n = n
        self.check_numeric()
        self.check_lims()

    def get(self, n=None):
        '''Method to get the random numbers

        Parameters
        ----------
        n : int, optional
            Number of random numbers to be generated. If passed in the callback
            of this function, it will override the object value.

        Returns
        -------
        out : ndarray
            1D array containing the generated random numbers
        '''
        if n is None and self.n is not None:
            n = self.n
        return random.uniform(self.a, self.b, n)


class Lognormal(RandomNumber):
    '''Class to create Lognormally distributed random numbers'''
    def __init__(self, mu, std, n=None):
        '''
        Class to create Lognormally distributed random numbers

        Parameters
        ----------
        mu : float
            Mean value of the distribution
        std : float
            Standard deviation of the distribution
        n : int, optional
            Number of random numbers to be generated

        '''
        self.mu = mu
        self.std = std
        self.n = n

        self.check_numeric()
        return

    def get(self, n=None):
        '''Method to get the random numbers

        Parameters
        ----------
        n : int, optional
            Number of random numbers to be generated. If passed in the callback
            of this function, it will override the object value.

        Returns
        -------
        out : ndarray
            1D array containing the generated random numbers
        '''
        if n is None and self.n is not None:
            n = self.n
        return random.lognormal(self.mu, self.std, n)


class Truncnormal(RandomNumber):
    '''Class to create truncated Normally distributed random numbers'''

    def __init__(self, mu, std, a=-np.inf, b=np.inf, n=None):
        '''
        Class to create truncated Normally distributed random numbers

        Parameters
        ----------
        mu : float
            Mean value of the distribution
        std : float
            Standard deviation of the distribution
        a : float, optional
            Lower limit of the distribution
        b : float, optional
            Upper limit of the distribution
        n : int, optional
            Number of random numbers to be generated

        '''
        self.mu = mu
        self.std = std
        self.a = a
        self.b = b
        self.n = n
        self.check_numeric()
        self.check_lims()

        return

    def get(self, n=None):
        '''Method to get the random numbers

        Parameters
        ----------
        n : int, optional
            Number of random numbers to be generated. If passed in the callback
            of this function, it will override the object value.

        Returns
        -------
        out : ndarray
            1D array containing the generated random numbers
        '''
        if n is None and self.n is not None:
            n = self.n
        return _tn(self.mu, self.std, self.a, self.b, n)

#
# if __name__ == '__main__':
#    print(Normal(0, 1).get(2))
#    print(Beta(1, 2).get(2))
#    print(Uniform(0, 1).get(2))
#    print(Lognormal(1, 2).get(2))
#    print(Truncnormal(1, 1, 0, 2).get(2))
#    print(Normal(1, 1).__getstate__())
