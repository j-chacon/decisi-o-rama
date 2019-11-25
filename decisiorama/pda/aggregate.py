# -*- coding: utf-8 -*-
""" Aggregate Module

This module contains a collection of functions for utility aggregation.

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
OFFSET = 1e-6

@np.vectorize
def _rerange(utils, offset=OFFSET):
    '''Re-range utilities so they are in the open interval (0,1)'''
    return utils*(1.0 - 2.0*offset) + offset


def _dimcheck(utils, w):
    '''Check the dimension consistency of inputs and weights'''
    if not utils.ndim == 2:
        msg = ('The dimensions of utils have to be (1, ) or (2, ) '
               'got {0}'.format(utils.ndim))
        raise ValueError(msg)

    if w is None:
        w = np.ones(utils.shape[1]) / utils.shape[1]

    elif callable(w[0]):  # check if its an iterable with a generator
        w = np.array([wi() for wi in w])
        w = w / np.sum(w, axis=0)

    if w.ndim == 1:
        if utils.shape[1] != w.shape[0]:
            msg = ('Weights and solutions do not match. The shape of solutions'
                   ' is {0} and of weights is {1}'.format(utils.shape[1],
                                                          w.shape)
                   )
            raise ValueError(msg)

    elif w.ndim == 2:
        if utils.shape != w.shape:
            msg = ('Weights and solutions do not match. The shape of '
                   'solutions is {0} and of weights is {1}'.format(utils.shape,
                                                                   w.shape)
                   )
            raise ValueError(msg)


def _w_normalize(w):
    '''Normalise the weights so the um is equal to 1'''
    if w.ndim == 1:
        w[:] = w / np.sum(w, axis=0)
    else:
        w[:] = np.array([wi / np.sum(wi) for wi in w])


def additive(utils, w, w_norm=True, *args, **kwargs):
    '''Additive utility aggregation function

    Aggregate preferences using a weighted average

    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The
        dimensions corresponds to the number of random samples (n) and the
        number of utilities (u)
    w : ndarray [u], [n, u]
        Array with the provided weights to each of the utilities. If passed
        as a 1D-array, the same weights are used for of all the random samples.
         In case it is a 2D-array, w requires the same dimensions as `utils`
    w_norm : Bool, optional
        If True, the sum of the weights will be equal to 1

    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        utils = np.array([0.0, 1.0])
        w = np.array([0.8, 0.2])
        print(additive(s,w))

        >>> [0.2]

        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([0.8, 0.2])
        print(additive(s,w))

        >>> [0.2 0.8 0.5]

        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([[0.8, 0.2],
                      [0.8, 0.2],
                      [0.8, 0.2]])
        print(additive(s,w))

        >>> [0.2 0.8 0.5]

    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1, -1])

    _dimcheck(utils, w)
    if w_norm:
        _w_normalize(w)

    if w.shape == utils.shape:
        out = np.sum(utils * w, axis=1)
    else:
        out = np.dot(utils, w)

    return out


def cobb_douglas(utils, w, w_norm=True, *args, **kwargs):
    '''Cobb-Douglas utility aggregation function

    Aggregate preferences using the cobb-douglas aggregation function. This
    method is also known as the weighted geometric average

    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The
        dimensions corresponds to the number of random samples (n) and the
        number of utilities (u)
    w : ndarray [u], [n, u]
        Array with the provided weights to each of the utilities. If passed
        as a 1D-array, the same weights are used for of all the random samples.
         In case it is a 2D-array, w requires the same dimensions as `utils`
    w_norm : Bool, optional
        If True, the sum of the weights will be equal to 1

    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        utils = np.array([0.0, 1.0])
        w = np.array([0.8, 0.2])
        print(cobb_douglas(utils, w))

        >>> [0.]

        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([0.8, 0.2])
        print(cobb_douglas(utils, w))

        >>> [0. 0. 0.5]

        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([[0.8, 0.2],
                      [0.8, 0.2],
                      [0.8, 0.2]])
        print(cobb_douglas(utils, w))

        >>> [0. 0. 0.5]

    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1, -1])

    _dimcheck(utils, w)
    if w_norm:
        _w_normalize(w)

    return np.prod(utils**w, axis=1)


def mix_linear_cobb(utils, w, pars=[0.5, ], w_norm=True, *args, **kwargs):
    '''to be deprecated'''
    if callable(pars[0]):
        alpha = pars[0]()
    else:
        alpha = pars[0]
    add_model = additive(utils, w, w_norm)
    cd_model = cobb_douglas(utils, w, w_norm)
    return alpha*(add_model) + (1.0 - alpha)*cd_model


def reverse_harmonic(utils, w, w_norm=True, *args, **kwargs):
    '''Reverse harmonic utility aggregation function

    Aggregate preferences using the cobb-douglas aggregation function. This
    method is also known as the weighted geometric average

    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The
        dimensions corresponds to the number of random samples (n) and the
        number of utilities (u)
    w : ndarray [u], [n, u]
        Array with the provided weights to each of the utilities. If passed
        as a 1D-array, the same weights are used for of all the random samples.
         In case it is a 2D-array, w requires the same dimensions as `utils`
    w_norm : Bool, optional
        If True, the sum of the weights will be equal to 1

    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        utils = np.array([0.0, 1.0])
        w = np.array([0.8, 0.2])
        print(reverse_harmonic(utils, w))

        >>> [1.]

        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([0.8, 0.2])
        print(reverse_harmonic(utils, w))

        >>> [1. 1. 0.5]

        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([[0.8, 0.2],
                      [0.8, 0.2],
                      [0.8, 0.2]])
        print(reverse_harmonic(utils, w))

        >>> [1. 1. 0.5]

    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1, -1])

    _dimcheck(utils, w)
    if w_norm:
        _w_normalize(w)

    return 1.0 - 1.0 / (np.sum(w / (1.0 - utils), axis=1))


def reverse_power(utils, w, alpha, w_norm=True, *args, **kwargs):
    '''Reverse power utility aggregation function

    Aggregate preferences using the reverse power aggregation function.

    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The
        dimensions corresponds to the number of random samples (n) and the
        number of utilities (u)
    w : ndarray [u], [n, u]
        Array with the provided weights to each of the utilities. If passed
        as a 1D-array, the same weights are used for of all the random samples.
        In case it is a 2D-array, w requires the same dimensions as `utils`
    w_norm : Bool, optional, default True
        If True, the sum of the weights will be equal to 1
    alpha : float, ndarray [n], default 1.0
        power coefficient. If passed as a float, the values will remain the
        same over the whole computation. Otherwise, it is possible to pass a
        vector with a value for each random sample

    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        utils = np.array([0.0, 1.0])
        w = np.array([0.8, 0.2])
        alpha = 1.0
        print(reverse_power(utils, w, alpha))

        >>> [0.2]

        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([0.8, 0.2])
        alpha = np.array([1.0, 1.0, 1.0])
        print(reverse_power(utils, w, alpha))

        >>> [0.2 0.8 0.5]

        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([[0.8, 0.2],
                      [0.8, 0.2],
                      [0.8, 0.2]])
        alpha = np.array([1.0, 1.0,1.0])
        print(reverse_power(utils, w, alpha))

        >>> [0.2 0.8 0.5]

    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1, -1])

    _dimcheck(utils, w)
    if w_norm:
        _w_normalize(w)

    if type(alpha) is np.ndarray:
        if alpha.ndim == 1:
            alpha = np.tile(alpha, (utils.shape[1], 1)).T
#            print('alpha: {0}'.format(alpha))
            out = 1.0 - np.power(np.sum(np.power(w*(1.0 - utils), alpha),
                                        axis=1), 1.0/alpha[:, 0])
        else:
            _msg = ('alpha has to be scalar or 1D array, '
                    'got {0}'.format(alpha.ndim))
            raise ValueError(_msg)
    else:
        out = 1.0 - np.power(np.sum(np.power(w*(1.0 - utils), alpha),
                                    axis=1), 1.0/alpha)
    return out


def multiplicative(utils, w, w_norm=True, *args, **kwargs):
    raise NotImplementedError('This method has not been implemented yet')


def split_power(utils, w, alpha, s, w_norm=True, *args, **kwargs):
    '''Split power utility aggregation function

    Aggregate preferences using the split power aggregation function.

    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The
        dimensions corresponds to the number of random samples (n) and the
        number of utilities (u)
    w : ndarray [u], [n, u]
        Array with the provided weights to each of the utilities. If passed
        as a 1D-array, the same weights are used for of all the random samples.
        In case it is a 2D-array, w requires the same dimensions as `utils`
    alpha : float, ndarray[n]
        Alpha parameter of the power function. In case a float value is used,
        it will be constant for all of the random samples
    s : float, ndarray[n]
        s parameter of the power function. In case a float value is used,
        it will be constant for all of the random samples
    w_norm : Bool, optional
        If True, the sum of the weights will be equal to 1

    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        utils = np.array([0.0, 1.0])
        w = np.array([0.8, 0.2])
        alpha = 1.0
        s = 1.0
        print(split_power(utils, w, alpha, s))

        >>> [0.2]

        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([0.8, 0.2])
        alpha = np.array([1.0, 1.0, 1.0])
        s = 1.0
        print(split_power(utils, w, alpha, s))

        >>> [0.2 0.8 0.5]

        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([[0.8, 0.2],
                      [0.8, 0.2],
                      [0.8, 0.2]])
        alpha = np.array([1.0, 1.0, 1.0])
        s = np.array([1.0, 1.0, 1.0])
        print(split_power(utils, w, alpha, s))

        >>> [0.2 0.8 0.5]

        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([[0.8, 0.2],
                      [0.8, 0.2],
                      [0.8, 0.2]])
        alpha = 1.0
        s = np.array([1.0, 1.0, 1.0])
        print(split_power(utils, w, alpha, s))

        >>> [0.2 0.8 0.5]

    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1, -1])

    _dimcheck(utils, w)
    if w_norm:
        _w_normalize(w)

    @np.vectorize
    def _g(u, s, alpha):
        if u <= s:
            out = s*(u/s)**alpha
        else:
            out = 1.0 - (1.0 - s)*((1.0 - u)/(1.0 - s))**alpha
        return out

    @np.vectorize
    def _g_inv(u, s, alpha):
        if u <= s:
            out = s*(u/s)**(1.0/alpha)
        else:
            out = 1.0 - (1.0 - s)*((1.0 - u)/(1.0 - s))**(1.0/alpha)
        return out

    if type(alpha) is np.ndarray:
        if alpha.ndim == 1:
            _alpha = np.tile(alpha, (utils.shape[1], 1)).T

        else:
            _msg = ('alpha has to be scalar or 1D array, '
                    'got {0}'.format(alpha.ndim))
            raise ValueError(_msg)
    else:
        _alpha = alpha

    if type(s) is np.ndarray:
        if s.ndim == 1:
            _s = np.tile(s, (utils.shape[1], 1)).T

        else:
            _msg = ('s has to be scalar or 1D array, '
                    'got {0}'.format(alpha.ndim))
            raise ValueError(_msg)
    else:
        _s = s

    out = _g_inv(np.sum(w*_g(utils, _s, _alpha), axis=1), s, alpha)
    return out


def harmonic(utils, w, w_norm=True, rerange=False, *args, **kwargs):
    '''Harmonic utility aggregation function

    Aggregate preferences using the reverse power aggregation function.

    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The
        dimensions corresponds to the number of random samples (n) and the
        number of utilities (u)
    w : ndarray [u], [n, u]
        Array with the provided weights to each of the utilities. If passed
        as a 1D-array, the same weights are used for of all the random samples.
        In case it is a 2D-array, w requires the same dimensions as `utils`
    w_norm : Bool, optional
        If True, the sum of the weights will be equal to 1
    rerange : Bool, optional
        Changes the range of utils to be in the open interval (0,1), defined
        by the offset value (defined at a library level as OFFSET, 1e-6).
        By default is set to False.

    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        utils = np.array([0.0, 1.0])
        w = np.array([0.8, 0.2])
        print(harmonic(utils, w, rerange=True))

        >>> [1.24999969e-06]

        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([0.8, 0.2])
        print(harmonic(utils, w, rerange=True))

        >>>[1.24999969e-06 4.99998000e-06 5.00000000e-01]

        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([[0.8, 0.2],
                      [0.8, 0.2],
                      [0.8, 0.2]])
        print(harmonic(utils, w, rerange=True))

        >>>[1.24999969e-06 4.99998000e-06 5.00000000e-01]

        utils = np.array([0.0, 1.0])
        w = np.array([0.8, 0.2])
        print(harmonic(utils, w, rerange=False))

        >>> [0.]

        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([0.8, 0.2])
        print(harmonic(utils, w, rerange=False))

        >>> [0.  0.  0.5]

        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([[0.8, 0.2],
                      [0.8, 0.2],
                      [0.8, 0.2]])
        print(harmonic(utils, w, rerange=False))

        >>> [0.  0.  0.5]

    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1, -1])

    _dimcheck(utils, w)
    if w_norm:
        _w_normalize(w)
    if rerange:
        utils = _rerange(utils, OFFSET)
    return 1.0 / np.sum(w/utils, axis=1)


def maximum(utils, *args, **kwargs):
    '''Maximum utility aggregation function

    Aggregate preferences using the maximum aggregation function.

    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The
        dimensions corresponds to the number of random samples (n) and the
        number of utilities (u)

    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        utils = np.array([[0.0, 1.0],
                          [1.0, 0.0],
                          [0.5, 0.5]])
        print(maximum(utils))

        >>> [1.  1.  0.5]

    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1, -1])

    return np.max(utils, axis=1)


def minimum(utils, *args, **kwargs):
    '''Minimum utility aggregation function

    Aggregate preferences using the minimum aggregation function.

    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The
        dimensions corresponds to the number of random samples (n) and the
        number of utilities (u)

    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        utils = np.array([[0.0, 1.0],
                          [1.0, 0.0],
                          [0.5, 0.5]])
        print(minimum(utils))

        >>> [0.  0.  0.5]

    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1, -1])

    return np.min(utils, axis=1)


def mix(utils, w, methods, w_methods, mix_fun, w_norm=True, methods_args=None,
        mix_args=None, *args, **kwargs):
    '''mixed utility aggregation function

    Aggregate preferences using a mix of aggregation functions.

    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The
        dimensions corresponds to the number of random samples (n) and the
        number of utilities (u)
    w : ndarray [u], [n, u]
        Array with the provided weights to each of the utilities. If passed
        as a 1D-array, the same weights are used for of all the random samples.
        In case it is a 2D-array, w requires the same dimensions as `utils`
    methods : list [m]
        a list of functions that will create each individual member of the
        model mixture
    w_methods : ndarray [m], [n, m]
        An array for the weights that will be used to mix each of the methods
    mix_fun : function
        Function that will be used to aggregate each of the members of the
        methods
    w_norm : Bool, optional
        If True, the sum of the weights will be equal to 1

    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([[0.8, 0.2],
                      [0.8, 0.2],
                      [0.8, 0.2]])
        methods = [cobb_douglas,
                   additive,]
        w_methods = np.array([0.5, 0.5])
        mix_fun = additive
        print(mix(utils, w, methods, w_methods, mix_fun))

        >>> [0.1 0.4 0.5]

        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([[0.8, 0.2],
                      [0.8, 0.2],
                      [0.8, 0.2]])
        methods = [cobb_douglas,
                   split_power,]
        methods_args = [{},
                        dict(alpha = 1.0, s = 1.0)]
        w_methods = np.array([0.5, 0.5])
        mix_fun = additive
        print(mix(utils, w, methods, w_methods, mix_fun,
                  methods_args=methods_args))
        >>> [0.1 0.4 0.5]


        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([[0.8, 0.2],
                      [0.8, 0.2],
                      [0.8, 0.2]])
        methods = [cobb_douglas,
                   additive,]
        mix_args = dict(alpha = 1.0, s = 1.0)
        w_methods = np.array([0.5, 0.5])
        mix_fun = split_power
        print(mix(utils, w, methods, w_methods, mix_fun, mix_args=mix_args))
        #>>> [0.1 0.4 0.5]


        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([[0.8, 0.2],
                      [0.8, 0.2],
                      [0.8, 0.2]])
        methods = [cobb_douglas,
                   additive,]
        mix_args = dict(alpha = 1.0, s = 1.0)
        w_methods = np.array([[0.5, 0.5],
                              [0.5, 0.5],
                              [0.5, 0.5]])
        mix_fun = split_power
        print(mix(utils, w, methods, w_methods, mix_fun, mix_args=mix_args))
        >>> [0.1 0.4 0.5]


        utils = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]])
        w = np.array([[0.8, 0.2],
                      [0.8, 0.2],
                      [0.8, 0.2]])
        methods = [cobb_douglas,
                   additive,]
        mix_args = dict(alpha = np.array([1.0, 1.0, 1.0]), s = 1.0)
        w_methods = np.array([[0.5, 0.5],
                              [0.5, 0.5],
                              [0.5, 0.5]])
        mix_fun = split_power
        print(mix(utils, w, methods, w_methods, mix_fun, mix_args=mix_args))
        >>> [0.1 0.4 0.5]

    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1, -1])

    if w_methods.ndim == 1:
        _dim_w_methods = w_methods.shape[0]
    elif w_methods.ndim == 2:
        _dim_w_methods = w_methods.shape[1]

    _dimcheck(utils, w)

    if len(methods) != _dim_w_methods:
        _msg = ('length of methods ({0}) and w_methods ({1}) are not '
                'the same'.format(len(methods), len(w_methods))
                )
        raise ValueError(_msg)

    if w_norm:
        _w_normalize(w)
        _w_normalize(w_methods)

    if methods_args is None:
        methods_args = [{}, ]*len(methods)

    if mix_args is None:
        mix_args = {}

    agg_util = [m(utils, w, **methods_args[i]) for i, m in enumerate(methods)]
    agg_util = np.array(agg_util).T
    return mix_fun(agg_util, w_methods, **mix_args)


def bonferroni(utils, w=None, w_norm=False):
    raise NotImplementedError('Not implemented yet')


def power(utils, w=None, w_norm=False):
    raise NotImplementedError('Not implemented yet')


def choquet(utils, w=None, w_norm=False):
    raise NotImplementedError('Not implemented yet')


def sugeno(utils, w=None, w_norm=False):
    raise NotImplementedError('Not implemented yet')

if __name__ == '__main__':
    #%%
    import matplotlib.pyplot as plt
    x = np.array([np.linspace(0, 1, 10), np.linspace(0, 1, 10)]).T
    
    w = np.linspace(0, 1, 10)
    w = np.array([w, 1-w]).T
    
    X1, X2 = np.meshgrid(x[:,0], x[:,1])
    X = np.array([X1.flatten(), X2.flatten()]).T
    
    W1, W2 = np.meshgrid(w[:,0], w[:,1])
    W = np.array([W1.flatten(), W2.flatten()]).T
    
    funs = [additive, cobb_douglas, harmonic, reverse_harmonic, 
            split_power, reverse_power, maximum, minimum]
    labs = ['Additive', 'Cobb-Douglas', 'Harmonic', 'Reverse Harmonic',
            'Split Power', 'Reverse power', 'Maximum', 'Minimum']
    plt.figure(figsize=[7, 9])
    _iix = 0
    for _i, f in enumerate(funs):
        for _j in range(6):
            _iix += 1 
            _w = _j/10.0
            res = f(X, np.array([_w, 1.0 - _w]), w_norm=False, alpha=1.0, s=1.0)
            res = np.reshape(res, [10,10])
            
            plt.subplot(len(funs), 6, _iix)
#            plt.imshow(res, interpolation='bilinear', origin='lower', extent=(0,1,0,1))
            pl = plt.contour(X1, X2, res, colors='k')
            plt.clabel(pl, inline=1, fontsize=3)
            
            if _i != 7:
                plt.xticks(())
            else:
                plt.xlabel('[{0}, {1}]'.format(_w, 1.0 - _w), fontsize=8)
                
            if _j != 0:
                plt.yticks(())
            else:
                plt.ylabel(labs[_i], fontsize=8)
                
#            plt.tick_params()
#            plt.title('{0}\nw=[{1}, {2}]'.format(labs[_i], _j/10.0, 1.0 - _j/10.0), fontsize=8)
#            plt.xlabel('X1')
#            plt.ylabel('X2')
    plt.tight_layout()
    
#    for f in funs:
    plt.savefig(r"\\tudelft.net\staff-homes\C\jchaconhurtado\My Documents\bw_agg.png", dpi=700)
        
    