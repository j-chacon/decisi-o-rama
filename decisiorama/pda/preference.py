# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:05:19 2019

@author: jchaconhurtado
"""
import numpy as np
from . import utility
from . import aggregate
#from . import ranker
from .. import utils

_num_types = (int, float, np.float, np.float16, np.float32, np.float64, 
                 np.int, np.int16, np.int32, np.int64)

def _isnumeric(x):
    return isinstance(x, _num_types)

class Objective():
    ''' Class to create a node in the hierarchy tree
    
    This is the base class for structuring each of the nodes for the 
    hierarchical aggregation. In the hierarchy, all of the nodes are instances 
    of the Objective class.
    
    '''
    def __init__(self, name, w, alternatives, obj_min, obj_max, n=100, 
                 utility_func=utility.exponential, utility_pars=0.0, 
                 aggregation_func=aggregate.additive, aggregation_pars=[0.5,], 
                 maximise=True):
        '''
        Parameters
        ----------
        
        name : str
            Name of the objective. This will be used to keep track of the nodes
        w : float, ndarray or ri instance
            Weight of the objective in the hierarchical aggregation.
        alternatives : list
            List containing the consequences of each action in the objective. This 
            only have an impact on leaf nodes, and will be override in parent 
            nodes.
        obj_min : float
            Minimum value of the objective. Is used to create the utility function.
        obj_max : float, optional
            Maximum value of the objective. Is used to create the utility function.
        n : float
            Number of random samples to be used in the uncertainty analysis.
        utility_function : func
            Function that converts the values of the objectives into utilities
        utility_pars : dict
            Dictionary with extra inputs to the utility function. Only used if it 
            is a leaf node.
        aggregation_function : func
            Aggregation function for this particular objective. Only used if it is 
            not a leaf node.
        aggregation_pars : dict
            Additional parameters to pass to the aggregation function
        maximise : Bool
            Indicates if the optimal of the objective function is to maximise. In 
            case the optimal value is its minimum, set to False
        '''
        
        if type(name) is not str:
            msg = 'name has to be string, got {0}'.format(type(name))
            raise TypeError(msg)
        self.name = name
        
        if not _isnumeric(obj_min):
            msg = ('obj_min is not of numeric type. got ' 
                   '{0}'.format(type(obj_min)))
            raise TypeError(msg)
        self.obj_min = obj_min
        
        if not _isnumeric(obj_max):
            msg = ('obj_max is not of numeric type. got '
                   '{0}'.format(type(obj_max)))
            raise TypeError(msg)
        self.obj_max = obj_max
        
        self.alternatives = alternatives  # for every action a generator
        
        if type(maximise) is not bool:
            msg = ('maximise is not of bool type. got '
                   '{0}'.format(type(maximise)))
            raise TypeError(msg)
        self.maximise = maximise
        
        self.utility_pars = utility_pars
        
        if not callable(utility_func):
            msg = 'utility_func is not a callable type'
            raise TypeError(msg)
        self.utility_func = utility_func
        
        if not callable(aggregation_func):
            msg = 'aggregation_func is not a callable type'
            raise TypeError(msg)
        self.aggregation_func = aggregation_func
        
        self.aggregation_pars = aggregation_pars
        
        self.w = w
        
        if type(n) is not int:
            msg = 'n should be int, got {0}'.format(type(n))
            raise TypeError()
        self.n = n
        
        self.children = []
        self.all_children = [name, ]
        return
    
    def __getstate__(self,):
        '''Function to get state of current object. It is pickable'''
        state = dict(
            name = self.name,
            obj_min = self.obj_min,
            obj_max = self.obj_max,
            alternatives = self.alternatives,
            maximise = self.maximise,
            utility_pars = self.utility_pars,
            utility_func = self.utility_func,
            aggregation_func = self.aggregation_func,
            aggregation_pars = self.aggregation_pars,
            children = self.children,
            w = self.w,
            n = self.n,
            all_children = self.all_children
            )
        return state
    
    def __setstate__(self, state):
        '''Function to set state of current object. It is pickable'''
        self.name = state['name']
        self.obj_min = state['obj_min']
        self.obj_max = state['obj_max']
        self.alternatives = state['alternatives']
        self.maximise = state['maximise']
        self.utility_pars = state['utility_pars']
        self.utility_func = state['utility_func']
        self.aggregation_func = state['aggregation_func']
        self.aggregation_pars = state['aggregation_pars']
        self.children = state['children']
        self.w = state['w']
        self.n = state['n']
        self.all_children = state['all_children']
        return
    
    def add_children(self, children):
        '''This is the method to add children nodes
        
        This method is used to create a child node in the hierarchy tree
        
        Parameters
        ----------
        
        children : objective
            An instance of the Objective node is passed.
        '''
        # check if children is the same as the current node (no self reference)
        if self.name == children.name:
            msg = 'It is not possible to have a self reference'
            raise AttributeError(msg)
        
        if (children.name in self.all_children or 
            self.name in children.all_children):
            msg = 'Not possible to have a circular reference'
            raise AttributeError(msg)
        
        self.children.append(children)
        for ci in children.all_children:
            self.all_children.append(ci)
        
        return

    def get_value(self, x):
        ''' get the attribute utlity based on the portfolio of actions
        
        This function is used to calculate the value of the attribute 
        (objective), for a given portfolio. If it is a leaf node, the values 
        must be provided to the object, otherwise it is calculated from the 
        hierarchical aggregation.
        
        Parameters
        ----------
        x : 1D array
            A binary vector that represent the portfolio of actions. 1 for 
            done, and 0 for no action
        
        Returns
        -------
        out : ndarray
            Utility value of the attributes
            
        '''
        
        if not hasattr(x, '__iter__'):
            msg = 'x is not an iterable'
            raise TypeError(msg)
        x = np.array(x)
        
        if x.ndim != 1:
            msg = 'Number of dimensions of x is not 1. got {0}'.format(x.ndim)
            raise AttributeError(msg)
        
        if self.children == []:
            
            if callable(self.alternatives):  # using a solution generator
                _sols = self.alternatives(self.n)                    
            # Check if its a list of callables
            elif callable(self.alternatives[0]):  
                _sols = np.array([r(self.n) for r in self.alternatives]).T                    
            else:  # using a pre-rendered list
                _sols = self.alternatives
            
            _sols *= x
            _sols = _sols.T

            # rank-normalise the alternatives
            _sols = (_sols - self.obj_min) / (self.obj_max - self.obj_min)
            
            if self.maximise is False:
                _sols = 1.0 - _sols

            # clip the alternatives (may be unnecessary)
            _sols = np.clip(_sols, 0.0, 1.0)

            # apply the utility function to the actions and add up
#            value = np.zeros([x.size, self.n])

            if callable(self.utility_pars[0]):  # Case using a generator
                ut_pars = [ut(self.n) for ut in self.utility_pars]
            elif hasattr(self.utility_pars[0], '__iter__'):  # Check iterable
                ut_pars = [ut[:](self.n) for ut in self.utility_pars]
            else:
                ut_pars = self.utility_pars   
            
            value = self.utility_func(_sols, ut_pars)
            value = np.sum(value, axis=0)
        
        else:
            # Calculate the utility for each children
            _temp_util = np.array([c.get_value(x) for c in self.children]).T
            
#            value = np.zeros(self.n)

            # get random weights
            if callable(self.children[0].w):  # Case using a generator
                _w = np.array([c.w(self.n) for c in self.children]).T
            elif hasattr(self.children[0].w, '__iter__'):  # Check iterable
                _w = np.array([c.w[:](self.n) for c in self.children]).T
            else:
                _w = np.array([c.w for c in self.children]).T

            value = self.aggregation_func(utils=_temp_util, 
                                          w=_w, 
                                          pars=self.aggregation_pars, 
                                          w_norm=True)
        return value
    
def hierarchy_smith(h_map, prob):
    '''Function to creeate a hierarchy in the problem object
    
    This function heps in building the hierarchies, by helping in the 
    definition of child nodes. This can also be done manually for each 
    objective
    
    Parameters
    ----------
    h_map : dict
        Dictionary containing the child nodes. The key of the dictionary has 
        to be the parent node. The keys in the maps have to be consistent with 
        the keys used in the problem (prob) definition.
    prob : dict
        Dictionary containing all of the nodes. This will be mutated as a 
        result of this operation
    '''
    for node in h_map.keys():
        for child in h_map[node]:
            prob[node].add_children(prob[child])
    return

#%%
class Evaluator():
    ''' Function to pos process the utilsults of the problem
    
    The utilsults will be posprocessed here. This object may contain methods to 
    rank the solutions, assess the performance and do sensitivity analysis
    '''
    def __init__(self, portfolio, utils, labels=None):
        ''' Constructor of the Evaluator class
        
        Parameters
        ----------
        portfolio : ndarray
            Array with the portfolio of decisions. This can be seed as the 
            same portfolio of decisions that were used to calculate the 
            utilities of a given objective.
        utils : ndarray
            Array with resulting utilities for each of the portfolios passed 
            as inputs.
        '''
        
        self.portfolio = portfolio
        self.utils = utils
        self.n_sols = utils.shape[0]
        self.labels = labels
        self.functions = []
        self.minimize = []
        return
    
    def __getstate__(self):
        '''Makes the object pickleable'''
        state = dict(
                portfolio = self.portfolio,
                utils = self.utils,
                n_sols = self.n_sols,
                labels = self.labels,
                )
        return state
    
    def __setstate__(self, state):
        '''Makes the object pickleable'''
        self.portfolio = state['portfolio']
        self.utils = state['utils']
        self.n_sols = state['n_sols']
        self.labels = state['labels']
        return
    
    def add_function(self, function, minimize=True):
        '''
        add an objective function for evaluating the solutions
        
        Parameters
        ----------
        
        function : func
            function to process the a 2D vector containing the stochastic 
            utilities for each portfolio
        minimize : Bool
            Determines if the function is to be minimised. If False, it means 
            the function will be maximised. Default is True.
        '''
        
        self.minimize.append(minimize)
        if minimize:
            self.functions.append(function)
        else:
            _f = lambda sols : -function(sols)
            self.functions.append(_f)
            
    def _performance(self):
        '''Calculate the performance indicators'''        
        perf = np.zeros([len(self.functions), self.n_sols])
        for i, func in enumerate(self.functions):
            perf[i, :] = func(self.utils)            
        return perf
    
    def get_ranked_solutions(self):
        '''Get the pareto ranking of the solutions'''
        perf = self._performance().T
        return utils.pareto_front_i(perf, minimize=True)
    
    def get_weak_ranked_solutions(self, i=0):
        '''Get weakly ranked solutions'''
        perf = self._performance().T
        _temp = []
        for i in range(i+1):
            _temp.append(utils.pareto_front_i(perf, minimize=False, i=i))
        return np.array([item for sublist in _temp for item in sublist])
    
    def get_core_index(self, i=0):
        # get pf
        pf = self.get_weak_ranked_solutions(i)
        return np.mean(self.portfolio[pf], axis=0)

#%%
if __name__ == '__main__':
    print('None')
#    from numpy.random import beta, normal, uniform
#    
#    n = 3
#    def sol_generator():
#        return np.array([ # rehab
#            beta(9.0375, 4.0951),
#            beta(9.0375, 4.0951),
#            beta(19.0754,8.9788),
#            uniform(0,0),
#            uniform(0,0),
#            uniform(0,0),
#            beta(19.0754, 8.9788),
#            uniform(0,0),
#            normal(0.0438, 0.0162),
#            normal(0.0438, 0.0162),
#            uniform(0,0),
#                  ])
#    sol_generator()
#
#    x0 = np.zeros(11)
#    x1 = np.ones(11)        
#    rehab = objective(name='rehab', 
#                      label='Rehab', 
#                      obj_min=0.0, 
#                      obj_max=100.0, 
#                      w=0.5, alternatives=sol_generator, n=n, 
#                      utility_func=utility.exponential, 
#                      utility_pars=[0.01,],  
#                      aggregation_func=aggregate.mix_linear_cobb, 
#                      aggregation_pars=[0.5,], maximise=True)
#    
#    print(rehab.get_value(x0))
#    print(rehab.get_value(x1))