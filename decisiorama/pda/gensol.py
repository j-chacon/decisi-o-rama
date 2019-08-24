# -*- coding: utf-8 -*-
""" gensol Module

This module contains generator of solutions for the
"""

__author__ = "Juan Carlos Chacon-Hurtado"
__credits__ = ["Juan Carlos Chacon-Hurtado", "Lisa Scholten"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Juan Carlos Chacon-Hurtado"
__email__ = "j.chaconhurtado@tudelft.nl"
__status__ = "Development"
__last_update__ = "22-08-2019"


# import numpy as np
import itertools as it
from warnings import warn


class solgen():

    def __init__(self, n):
        '''
        Class to generate the full enumeration of solutions

        Parameters
        ----------

        n : int
            Number of variables to build
        '''
        self.n = n
        self.allgen = it.product([0, 1], repeat=n)
        self.subseq = []  # subsequent activities
        self.mut_exc = []  # mutually exclusive

    def add_subsequent(self, parent, sub):
        '''
        Add a subsequent activity. This means that the sub activity cannot be
        present without the parent.

        Parameters
        ----------
        parent : int
            index of the parent activity

        sub : int
            index of the subsequent activity

        Example
        -------

        All possible combinations of 2 elements are:
        [(1, 1), (1, 0), (0, 1), (0, 0)]

        if we add that the first activity is required for the second to happen

        add_subsequent(0, 1)
        [(1, 1), (1, 0), (0, 0)]

        '''

        if parent >= self.n:
            _msj = ('parent is out of bounds. Maximum value is {0} and \
                    got {1}'.format(self.n, parent))
            raise ValueError(_msj)

        if sub >= self.n:
            _msj = ('sub is out of bounds. Maximum value is {0} and \
                    got {1}'.format(self.n, sub))
            raise ValueError(_msj)

        _ac = [parent, sub]
        if _ac not in self.subseq:
            self.subseq.append(_ac)
        else:
            warn('Subsequent activity already added')

    def add_mutually_exclusive(self, a, b):
        '''
        add_mutually_exclusive

        Function to add a mutually exclusive solution

        Parameters
        ----------
        a : int
            index of the first mutually exclusive activity

        b : int
            index of the first mutually exclusive activity

        Example
        -------

        All possible combinations of 2 elements are:
        [(1, 1), (1, 0), (0, 1), (0, 0)]

        if we add mutually exclusivity between them

        add_mutually_exclusive(0, 1)
        [(1, 0), (0, 1), (0, 0)]
        '''

        if a >= self.n:
            _msj = ('a is out of bounds. Maximum value is {0} and \
                    got {1}'.format(self.n, a))
            raise ValueError(_msj)

        if b >= self.n:
            _msj = ('b is out of bounds. Maximum value is {0} and \
                    got {1}'.format(self.n, b))
            raise ValueError(_msj)

        _ac = [a, b]
        if _ac not in self.mut_exc or _ac[::-1] not in self.mut_exc:
            self.mut_exc.append(_ac)
        else:
            warn('Mutualy exclusive activity already added')

    def generate(self):
        '''
        generate

        Function to create a generator of the solutions

        Returns
        -------
        Generator of the solutions

        '''

         # generate 3 solutions
        _all_sols = it.product([0, 1], repeat=self.n)

        for i in _all_sols:
            _flag = True

            # Check for subsequent activities
            for si in self.subseq:
                if [i[si[0]], i[si[1]]] == [1,1]:
                    _flag = False
                    break

            # Check for mutually exclusive
            if _flag:
                for ei in self.mut_exc:
                    if [i[ei[0]], i[ei[1]]] == [1,1]:
                        _flag = False
                        break

            if _flag:
                yield i

sol = solgen(3)
sol.add_subsequent(0, 1)
sol.add_mutually_exclusive(1,2)
sol.add_mutually_exclusive(1,2)
[print(i) for i in sol.generate()]

