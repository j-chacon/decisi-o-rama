# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:05:19 2019

@author: jchaconhurtado
"""
import numpy as np
import utility
import aggregate
import ranker
import utils

class objective():
    
    '''all problems will become maximisation problems'''
    
    def __init__(self, 
                 name, 
                 w, 
                 results, 
                 obj_min=-np.inf, 
                 obj_max=np.inf, 
                 n=100, 
                 utility_func=utility.exponential, 
                 utility_pars=0.0, 
                 aggregation_func=aggregate.mix_linear_cobb, 
                 aggregation_pars=[0.5,], 
                 maximise=True,
                 chunks=None):
        '''intialisation method'''
        self.name = name
        self.obj_min = obj_min
        self.obj_max = obj_max
        self.results = results  # for every action a generator
        
        self.maximise = maximise
        self.utility_pars = utility_pars
        self.utility_func = utility_func
        self.aggregation_func = aggregation_func
        self.aggregation_pars = aggregation_pars
        self.children = []
        self.w = w
        self.chunks = chunks
        
        if chunks is None:
            self.n = n
        else:
            self.n = n//chunks
            
        return
    
    def add_children(self, children):
        '''method to add childrens'''
        self.children.append(children)
        
        return
    
    def get_value(self, x):
        '''normalise the results of the actions before adding up'''
        x = np.array(x)
        
        if self.children == []:
            
            if callable(self.results):  # using a solution generator
                _sols = self.results(self.n)                    
            elif callable(self.results[0]):  # Check if its a list of callables
                _sols = np.array([r(self.n) for r in self.results]).T                    
            else:  # using a pre-rendered list
                _sols = self.results
            
            _sols *= x
            _sols = _sols.T

            # rank-normalise the results
            _sols = (_sols - self.obj_min) / (self.obj_max - self.obj_min)
            
            if self.maximise is False:
                _sols = 1.0 - _sols

            # clip the results (may be unnecessary)
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
    def __init__(self, inps, res):
        self.inps = inps
        self.res = res
        self.n_sols = res.shape[0]
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
    
    
    
#ee = evaluator(inps, np.array(res))
#g = ee.get_ranked_solutions([ranker.mean, ranker.iqr])
#g = ee.get_weak_ranked_solutions([ranker.mean, ranker.iqr], 3)
#print(ee.get_core_index([ranker.iqr, ranker.mean], 0))
    
if __name__ == '__main__':
    from numpy.random import beta, normal, uniform
    
    n = 3
    def sol_generator():
        return np.array([ # rehab
            beta(9.0375, 4.0951),
            beta(9.0375, 4.0951),
            beta(19.0754,8.9788),
            uniform(0,0),
            uniform(0,0),
            uniform(0,0),
            beta(19.0754, 8.9788),
            uniform(0,0),
            normal(0.0438, 0.0162),
            normal(0.0438, 0.0162),
            uniform(0,0),
                  ])
    sol_generator()

    x0 = np.zeros(11)
    x1 = np.ones(11)        
    rehab = objective(name='rehab', 
                      label='Rehab', 
                      obj_min=0.0, 
                      obj_max=100.0, 
                      w=0.5, results=sol_generator, n=n, 
                      utility_func=utility.exponential, 
                      utility_pars=[0.01,],  
                      aggregation_func=aggregate.mix_linear_cobb, 
                      aggregation_pars=[0.5,], maximise=True)
    
    print(rehab.get_value(x0))
    print(rehab.get_value(x1))