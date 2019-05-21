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

class objective():
    '''
    The objective class constructor allows to 
    '''
    def __init__(self, 
                 name, 
                 w, 
                 alternatives, 
                 obj_min=-np.inf, 
                 obj_max=np.inf, 
                 n=100, 
                 utility_func=utility.exponential, 
                 utility_pars=0.0, 
                 aggregation_func=aggregate.mix_linear_cobb, 
                 aggregation_pars=[0.5,], 
                 maximise=True,
                 ):
        '''intialisation method'''
        if type(name) is not str:
            raise TypeError('name has to be string, got {0}'.format(type(name)))
        self.name = name
        
        if not _isnumeric(obj_min):
            raise TypeError('obj_min is not of numeric type. got {0}'.format(type(obj_min)))
        self.obj_min = obj_min
        
        if not _isnumeric(obj_max):
            raise TypeError('obj_max is not of numeric type. got {0}'.format(type(obj_max)))
        self.obj_max = obj_max
        
        self.alternatives = alternatives  # for every action a generator
        
        if type(maximise) is not bool:
            raise TypeError('maximise is not of bool type. got {0}'.format(type(maximise)))
        self.maximise = maximise
        
        self.utility_pars = utility_pars
        
        if not callable(utility_func):
            raise TypeError('utility_func is not a callable type')
        self.utility_func = utility_func
        
        if not callable(aggregation_func):
            raise TypeError('aggregation_func is not a callable type')
        self.aggregation_func = aggregation_func
        
        self.aggregation_pars = aggregation_pars
        
        self.w = w
        if type(n) is not int:
            raise TypeError('n should be int, got {0}'.format(type(n)))
        self.n = n
        
        self.children = []
        self.all_children = [name, ]
        return
    
    def __getstate__(self,):
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
        '''method to add childrens'''
        # check if children is the same as the current node (no self reference)
        if self.name == children.name:
            raise AttributeError('It is not possible to have a self reference')
        
        if children.name in self.all_children or self.name in children.all_children:
            raise AttributeError('Not possible to have a circular reference')
        
        self.children.append(children)
        for ci in children.all_children:
            self.all_children.append(ci)
        
        return

    def get_value(self, x):
        '''normalise the alternatives of the actions before adding up'''
        
        if not hasattr(x, '__iter__'):
            raise TypeError('x is not an iterable')
        x = np.array(x)
        
        if x.ndim != 1:
            raise AttributeError('Number of dimensions of x is not 1. got {0}'.format(x.ndim))
        
        if self.children == []:
            
            if callable(self.alternatives):  # using a solution generator
                _sols = self.alternatives(self.n)                    
            elif callable(self.alternatives[0]):  # Check if its a list of callables
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

            value = self.aggregation_func(sols=_temp_util, 
                                          w=_w, 
                                          pars=self.aggregation_pars, 
                                          w_norm=True)
        return value
    
def hierarchy_smith(h_map, prob):
    '''Mutates the prob dictionary'''
    for node in h_map.keys():
        for child in h_map[node]:
            prob[node].add_children(prob[child])
    return

class evaluator():
    def __init__(self, inps, res, labels=None):
        self.inps = inps
        self.res = res
        self.n_sols = res.shape[0]
        self.labels = labels
        return
    
    def __getstate__(self):
        state = dict(
                inps = self.inps,
                res = self.res,
                n_sols = self.n_sols,
                labels = self.labels,
                )
        return state
    
    def __setstate__(self, state):
        self.inps = state['inps']
        self.res = state['res']
        self.n_sols = state['n_sols']
        self.labels = state['labels']
        return
    
    def _performance(self, functions):
        '''Calculate the performance indicators'''        
        perf = np.zeros([len(functions), self.n_sols])
        for i, func in enumerate(functions):
            perf[i, :] = func(self.res)            
        return perf
    
    def get_ranked_solutions(self, functions):
        '''Get the pareto ranking of the solutions'''
        perf = self._performance(functions).T
        return utils.pareto_front_i(perf, minimize=False)
    
    def get_weak_ranked_solutions(self, functions, i=0):
        '''Get weakly ranked solutions'''
        perf = self._performance(functions).T
        _temp = []
        for i in range(i+1):
            _temp.append(utils.pareto_front_i(perf, minimize=False, i=i))
        return np.array([item for sublist in _temp for item in sublist])
    
    def get_core_index(self, functions, i=0):
        # get pf
        pf = self.get_weak_ranked_solutions(functions, i)
        return np.mean(self.inps[pf], axis=0)

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