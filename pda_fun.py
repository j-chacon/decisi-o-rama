# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:05:19 2019

@author: jchaconhurtado
"""
import numpy as np
import utility
import aggregate
from numpy.random import beta, normal, lognormal, uniform
from scipy.stats import truncnorm

# Define the generator of the tuncated normal distributions
def tn(mu, sigma, lower=-np.inf, upper=np.inf, size=None):
    out = truncnorm((lower - mu) / sigma, 
                    (upper - mu) / sigma, 
                    loc=mu, 
                    scale=sigma)
    return out.rvs(size=size)


class objective():
    '''all problems will become maximisation problems'''
    
    def __init__(self, name, w, results, obj_min=-np.inf, obj_max=np.inf, n=100, 
                 utility_func=utility.exponential, 
                 utility_pars=0.0, 
                 aggregation_func=aggregate.mix_linear_cobb, 
                 aggregation_pars=[0.5,], 
                 maximise=True):
        '''intialisation method'''
        self.name = name
#        self.label = label
        self.obj_min = obj_min
        self.obj_max = obj_max
        self.results = results  # for every action a generator
        self.n = n
        self.maximise = maximise
        self.utility_pars = utility_pars
        self.utility_func = utility_func
        self.aggregation_func = aggregation_func
        self.aggregation_pars = aggregation_pars
        self.children = []
        self.w = w
            
        return
    
    def add_children(self, children):
        '''method to add childrens'''
        self.children.append(children)
        
        return
    
    def get_value(self, x):
        '''normalise the results of the actions before adding up'''
        x = np.array(x)
        
        if self.children == []:
            # calculate the utility from the solutions
            
            # get the solutions for x
            _sols = np.zeros([x.size, self.n])
            if callable(self.results):  # using a solution generator
                for i in range(self.n):
                    _sols[:, i] = self.results() * x
                    
            elif callable(self.results[0]):  # Check if its a list of callables
                for i in range(self.n):
                    _sols[:, i] = np.array([r() for r in self.results])
                    
            else:  # using a pre-rendered list
                for i in range(self.n):
                    _sols[:, i] = self.results[:, i] * x
            
            # rank-normalise the results
            _sols = (_sols - self.obj_min) / (self.obj_max - self.obj_min)
            
            if self.maximise is False:
                _sols = 1.0 - _sols
            
            # clip the results (may be unnecessary)
            _sols = np.clip(_sols, 0.0, 1.0)
            
            # apply the utility function to the actions and add up
#            print(self.utility_func(_sols, self.utility_pars).shape)
            value = np.zeros([x.size, self.n])
            for i in range(self.n):
                if callable(self.utility_pars[0]):  # Case using a generator
                    ut_pars = [ut() for ut in self.utility_pars]
                elif hasattr(self.utility_pars[0], '__iter__'):  # Check if its iterable
                    ut_pars = [ut[i] for ut in self.utility_pars]
                else:
                    ut_pars = self.utility_pars    
#                print(_sols[:,i])
#                print(ut_pars)
                value[:, i] = self.utility_func(_sols[:, i], ut_pars)
                
            value = np.sum(value, axis=0)
        
        else:
            # Calculate the utility for each children
            _temp_util = np.array([c.get_value(x) for c in self.children]).T
            value = np.zeros(self.n)
            for i in range(self.n):  # atomic operation
                # get random weights
                if callable(self.children[0].w):  # Case using a generator
                    _w = np.array([c.w() for c in self.children])
                elif hasattr(self.children[0].w, '__iter__'):  # Check if its iterable
                    _w = np.array([c.w[i] for c in self.children])
                else:
                    _w = np.array([c.w for c in self.children])
#                print(_w)
                # make the utility aggregation
                value[i] = self.aggregation_func(sols=_temp_util[i, :], w=_w, 
                                         pars=self.aggregation_pars, w_norm=True)
        
        return value
if __name__ == '__main__':
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
    rehab = objective(name='rehab', label='Rehab', obj_min=0.0, obj_max=100.0, w=0.5,
                      results=sol_generator, n=n, utility_func=utility.exponential, 
                      utility_pars=[0.01,],  
                      aggregation_func=aggregate.mix_linear_cobb, 
                      aggregation_pars=[0.5,], maximise=True)
    #
    print(rehab.get_value(x0))
    print(rehab.get_value(x1))