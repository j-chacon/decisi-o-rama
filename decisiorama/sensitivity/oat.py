# -*- coding: utf-8 -*-
""" One-at-a-time sensitivity analysis

This module contains functions to carry out a one-at-a-time sensitivity
analysis.
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


def _sample_problem(x):
    '''This is the Binh and Korn function'''

    # adding constraint 1
    if (x[0] - 5.0)**2 + x[1]**2 > 25.0:
        f1, f2 = np.nan, np.nan

    # adding constraint 2
    elif (x[0] - 8.0)**2 + (x[1] + 3.0)**2 < 7.7:
        f1, f2 = np.nan, np.nan

    # calculate the OF
    else:
        f1 = 4.0*x[0]**2 + 4.0*x[1]**2
        f2 = (x[0] - 5.0)**2 + (x[1] - 5.0)**2

    return f1, f2


# Make perturbation matrix
def _perturbation(pars, dx):
    '''calculate the perturbation vector for a OAT analysis'''
    pars = np.array(pars)
    dx = np.array(dx)

    n_pars = pars.size
    n_pert = dx[0].size
    perturbed_pars = np.zeros([n_pars, n_pert])  # upper and lower bounds
    for i in range(n_pars):
        for j in range(n_pert):
            perturbed_pars[i, j] = pars[i] + dx[i, j]
    return perturbed_pars


def oat_sensitivity(func, pars, dx):
    '''one at the time sensitivity analysis

    Parameters
    ----------

    func : function
        Function that calculate the sensitivity objectives
    pars : 1D array
        Parameters around which the OAT sensitivity will be calculated. This
        vector contains all of the arguments to run the function
    dx : 2D array
        Matrix containing the perturbance for each of the variables.

    Returns
    -------
    perturbed_res : 3D array
        Matrix with the results of the model runs for each perturbation with
        dimensions of parameter, perturbation and objetive function
    delta : 3D array

    '''
    if np.any(dx == 0):
        raise ValueError('dx values cannot contain zero')

    pars = np.array(pars)
    dx = np.array(dx)

    # get the parameters
    n_pars = pars.size
    n_pert = dx[0].size

    perturbed_pars = _perturbation(pars, dx)
    centered_res = np.array(func(pars))
    n_of = centered_res.size

    perturbed_res = np.zeros([n_pars, n_pert, n_of])
    delta = np.zeros([n_pars, n_pert, n_of])
    for i in range(n_pars):
        for j in range(n_pert):
            if dx[i, j] == 0.0:
                perturbed_res[i, j, :] = centered_res
                delta[i, j, :] = None
            else:
                _pp = np.zeros_like(pars)
                _pp[:] = pars
                _pp[i] = perturbed_pars[i, j]
                perturbed_res[i, j, :] = np.array(func(_pp))
                _dy = (centered_res - perturbed_res[i, j])
                delta[i, j, :] = _dy / (dx[i, j])

    return perturbed_res, delta


def local_sensitivity(func, pars, vicinity=0.05, n_pert=10):
    '''

    Create a local sensitivity analysis around a fractional value of the
    centered parameter values (vicinity).

    Parameters
    ----------

    func : function
        Function that calculate the sensitivity objectives
    pars : 1D array
        Parameters around which the OAT sensitivity will be calculated. This
        vector contains all of the arguments to run the function
    vicinity : float
        Fraction of the parameter that will be used in generating the local
        sensitivity
    n_pert : int
        Number of points for each variable in the calculation of the local
        sensitivity

    Returns
    -------
    lsi : 1D array
        Local sensitivity index as a normalised value

    '''
    # Create the perturbations aroun the vicinity
    perts = np.zeros([pars.size, n_pert])

    if type(vicinity) is float:
        vicinity = np.ones(pars.size) * vicinity

    for i in range(pars.size):
        pert_width = pars[i]*vicinity[i]
        perts[i, :] = np.linspace(pars[i] - pert_width,
                                  pars[i] + pert_width,
                                  n_pert)

    # Make oat sensitivity analysis
    perturbed_res, delta = oat_sensitivity(func, pars, perts)

    # calculate slope of the regression line in the perturbed values
    lsi = np.zeros(pars.size)
    for i in range(pars.size):
        lsi[i] = np.polyfit(perts[i, :], perturbed_res[i, :, 0], deg=1)[0]

    # normalise the results
    lsi = lsi / np.nansum(np.abs(lsi))

    return lsi

#
# if __name__ == '__main__':
#    import matplotlib.pyplot as plt
#
#    pars = np.array([3.0, 1.0 , 1.0])
#    pars_label = ['x0','x1','x2']
#    of_label = ['OF 1','OF 2']
#
#    dx = np.array([[-0.1, -0.05, 0.05, 0.1],
#                   [-0.1, -0.05, 0.05, 0.1],
#                   [-0.1, -0.05, 0.05, 0.1]])
#
#    res = oat_sensitivity(_sample_problem, pars, dx)
#    n_pars, n_pert, n_of = res[1].shape
#
#    for obj in range(n_of):
#        plt.figure()
#        plt.ylabel(r'$\delta y / \delta x$')
#        plt.grid()
#        for par in range(n_pars):
#            plt.plot(dx[par], res[1][par, :, obj], 'o-',label=pars_label[par])
#        plt.title(of_label[obj])
#        plt.legend()
#        plt.show()
#
#    for obj in range(n_of):
#        plt.figure()
#        plt.grid()
#        for par in range(n_pars):
#            plt.plot(dx[par], res[0][par, :, obj], 'o-',label=pars_label[par])
#        plt.ylabel(of_label[obj])
#        plt.legend()
#        plt.show()
