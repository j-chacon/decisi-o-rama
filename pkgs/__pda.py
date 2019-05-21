# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from random import Random
import matplotlib.pyplot as plt
import matplotlib
# plt.style.use('ggplot')
plt.style.use('seaborn-dark')

import inspyred
import utils

from os.path import join
import sys
pkgsdir = join('.','pkgs')
sys.path.append(pkgsdir)

import dSALib as sa
from dSALib.sample.saltelli import sample as sobol_sampler
from  dSALib.analyze.sobol import analyze as sobol_analyzer
import plot_help as ph


class Problem():
    
    def __init__(self, func, x0):
        '''
        This is the Problem class method that will be used to serve as a 
        container 
        
        Parameters
        ----------
        
        func : function
            Process model. Takes as inputs the actions, and as outputs the 
            objectives
        x0 : vector 
            Initial vector to run the model
        '''
        
        # Save initial variables
        self.func = func
        self.x0 = x0
        
        # Run the model
        _temp_out = func(x0)
        
        # Get objective dimensions
        self.obj_dim = _temp_out.shape
    
    def gsa(self, n, bounds, save_sample=True):
        '''
        Function to make global sensitivity analysis
        
        Parameters
        ----------
        
        n : int
            Number of samples in the sensitivity analysis
        
        bounds : 2d_array [m, 2]
            Matrix with the lower and upper boundary conditions for each parameter
        '''
        # Make the problem
        _sa_prob = dict(num_vars = self.x0.shape[0],
                        names = ['x'+str(i) for i in range(self.x0.shape[0])],
                        bounds = bounds,
                        )
        
        # Make sample
        _sample = sa.sample.saltelli.sample(_sa_prob, n)
        if save_sample:
            self.sample = _sample
        
        # Run the model using mp
        
if __name__ == '__main__':
    def sample_problem(x):
        '''This is the Binh and Korn function'''
        
        
        # adding constraint 1
        if (x[0] - 5.0)**2 + x[1]**2 > 25.0:
            f1,f2 = np.nan, np.nan
        
        # adding constraint 2
        elif (x[0] - 8.0)**2 + (x[1] + 3.0)**2 < 7.7:
            f1,f2 = np.nan, np.nan
        
        # calculate the OF
        else:
            f1 = 4.0*x[0]**2 + 4.0*x[1]**2  
            f2 = (x[0] - 5.0)**2 + (x[1] - 5.0)**2
        
        return f1, f2
    
    
    # Define the sensitivity analysis problem        
    n_vars = 3
    vars_label = ['X0','X1','X2']
    n_objs= 2
    objs_label = ['OF 1','OF 2']
    sa_prob = dict(num_vars = n_vars,
                   names = ['x','y','z'],
                   bounds = [[0, 5.0],
                             [0, 3.0],
                             [0, 3.0]]
                   )        
    _var_out = ['S1', 'ST', 'S2']  # Variables to be shown as outputs
    n = 1000
    calc_second_order = True
    par_vals = sobol_sampler(sa_prob, n, calc_second_order=calc_second_order)
    res = np.array([sample_problem(pari) for pari in  par_vals])
    
    # # parse sensitivity indexes
    # sens = [sobol_analyzer(sa_prob, 
    #                        res[:,i], 
    #                        calc_second_order=calc_second_order) for i in range(res.shape[1])]
    # ss = [np.array([sensi[var] for sensi in sens]) for var in _var_out]
    # ss_conf = [np.array([sensi[var+'_conf'] for sensi in sens]) for var in _var_out]
    
    # plot sensitivity indexes
    # ph.plot_s1(ss[0], ss_conf[0], label='S1', of_label=None, var_label=None)
    # ph.plot_s1(ss[1], ss_conf[1], label='ST', of_label=None, var_label=None)
    # ph.plot_s2(ss[2], ss_conf[2], label='S2', of_label=None, var_label=None)
    
    #%% Get the solutions in the pareto front


    # def calc_fronts(M, minimize=True):
    #     '''function to calculate the pareto fronts'''
    #     if minimize is True:
    #         i_dominates_j = np.all(M[:,None] <= M, axis=-1) & np.any(M[:,None] < M, axis=-1)
    #     else:
    #         i_dominates_j = np.all(M[:,None] >= M, axis=-1) & np.any(M[:,None] > M, axis=-1)
    #     remaining = np.arange(len(M))
    #     fronts = np.empty(len(M), int)
    #     frontier_index = 0
    #     while remaining.size > 0:
    #         dominated = np.any(i_dominates_j[remaining[:,None], remaining], axis=0)
    #         fronts[remaining[~dominated]] = frontier_index
    
    #         remaining = remaining[dominated]
    #         frontier_index += 1
    #     return fronts
    fronts = utils.pareto_fronts(res, minimize=True)
    # xx = np.where(fronts == np.max(fronts))
    xx = np.where(fronts == 0)[0]  # Dominating set
    #%%
    
    for i in range(n_objs):
        for j in range(i+1, n_objs):
            # Analyse tradeoffs between OF
            plt.figure()
            plt.plot(res[:,i], res[:,j], '.')  # Make axis plot
            plt.plot(res[xx, i], res[xx, j], '.', color='darkorange')
            plt.xlabel(objs_label[i])
            plt.ylabel(objs_label[j])
            plt.grid()
            plt.show()
            
    #%% parallel plot
    plt.figure()
    plt.plot(res.T, '#a6a6a6')  # Make parallel plot outline
    plt.plot(res[xx].T, color='darkorange')             
    plt.xticks(range(n_objs), objs_label)
    plt.grid()             
    plt.show()
    
    #%%
    # plt.figure()  # Make variable-objective plot
    fig, axs = plt.subplots(n_objs, n_vars)
    
    for i in range(n_vars):
        for j in range(n_objs):
            axs[j, i].plot(par_vals[:, i], res[:, j], '.', alpha=0.3)
            axs[j, i].plot(par_vals[xx, i], res[xx, j], '.', 
               color='darkorange', alpha=0.3)
            # axs[j,i].xaxis.set_label(vars_label[i])
            # axs[j,i].yaxis.set_label(objs_label[j])
            axs[j, i].grid()
            if j != n_objs-1:
                axs[j, i].xaxis.set_tick_params(labelbottom=False)
            
            if j == n_objs-1:
                axs[j, i].xaxis.set_label_text(vars_label[i]) 
            
            if i != 0:
                axs[j, i].yaxis.set_tick_params(labelleft=False)    
            
            if i == 0:
                axs[j, i].yaxis.set_label_text(objs_label[j])    
    plt.tight_layout()

    
    #%%
    
    # Filter the results in both samples and results
    # filter_idxs = np.isfinite(res).reshape([-1,2])[:,0]
    # filter_res = res[filter_idxs]
    # filter_pars = par_vals[filter_idxs]
    # filtered = np.hstack([filter_pars, filter_res])
    # norm_filtered = ((filtered - np.min(filtered, axis=0)) / 
    #                  (np.max(filtered, axis=0) - np.min(filtered, axis=0)))
    
    joined = np.hstack([par_vals, res])
    norm_filtered = ((joined - np.nanmin(joined, axis=0)) / 
                     (np.nanmax(joined, axis=0) - np.nanmin(joined, axis=0)))
    
    # Normalise the inputs
    #%%
    
    plt.figure()
    plt.plot(norm_filtered.T, color='#a6a6a6')
    plt.plot(norm_filtered[xx].T, color='darkorange')
    plt.xticks(range(n_objs + n_vars), vars_label + objs_label)
    plt.show()
    
    #%%
    # Make correlation plot between optimal solutions
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            plt.figure()
            plt.plot(par_vals[xx, i], par_vals[xx, j], '.')
            plt.xlabel(vars_label[i])
            plt.ylabel(vars_label[j])
            plt.show()
    
    
    #%% Make histograms of optimal solutions
    for var in res[xx].T:
        plt.figure()
        plt.hist(var)
        plt.show()
    
    #%%
    # # Define the value model
    # w = [0.2, 0.8]
    # norm_w = np.array(w)/np.sum(w)
    
    # def val_model(of, w):
    #     # normalise weights
    #     w = np.array(w)/np.sum(w)
    #     return np.sum([of[i]*w[i] for i in range(len(w))])
    
    
    # # Solve optimisation problem and find pareto-optimal solutions
    # prng = Random()
    
    # # Define the problem
    # # Generator - Create a random vector (candidate)
    # def generator(random, args):
    #     return [random.uniform(bound[0], bound[1]) for bound in sa_prob['bounds']]
    
    # # Evaluator - computes the OF
    # def evaluator(candidates, args):
    #     fitness = []
    #     for cs in candidates:
    #         fit = sample_problem(cs)
    #         fitness.append(fit)
    #     return fitness
        
    # # Bounder - Make sure a candidate is within the plausible range
    # def bounder(candidate, args):
    #     for i, c in enumerate(candidate):
    #         candidate[i] = np.clip(c, 
    #                                 sa_prob['bounds'][i][0], 
    #                                 sa_prob['bounds'][i][1])
    #     return candidate
    # #%%
    
    # # Implement the optimisation algorithm
    # ea = inspyred.ec.DEA(prng)
    # ea.terminator = inspyred.ec.terminators.evaluation_termination
    # for i in range(5):  # Run for 5 rounds
    #     ea.evolve(generator=generator, 
    #               evaluator=evaluator, 
    #               pop_size=100, 
    #               bounder=bounder,
    #               maximize=False,
    #               max_evaluations=500)
    
    #     sols = np.array([soli.fitness for soli in ea.population])
    #     plt.scatter(sols[:,0],sols[:,1], label=(i+1)*500)
    # plt.legend()
    # # Get pareto-dominated solutions
    
    
