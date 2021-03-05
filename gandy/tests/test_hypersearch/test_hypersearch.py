"""Testing hyperparameter optimization with optuna"""
import numpy
import unittest
import unittest.mock

import optuna.trial

import gandy.optimization.optimization as opt

class TestSearchableSpace(unittest.TestCase):
    
    def test_class(self):
        """try all possible searchable spaces"""
        NAME = 'hypname'
        
        # uniform float
        spaces = [(2.0, 4.0), (2.0, 4.0, 'uniform')]
        for space in spaces:
            subject = opt.SearchableSpace(NAME, space)
            self.assertEqual(subject.name, NAME)
            self.assertEqual(subject.args, (2.0, 4.0))
            self.assertTrue(subject.func is optuna.trial.Trial.suggest_uniform)
        
        # loguniform float
        space = (3.0, 8.0, 'loguniform')
        subject = opt.SearchableSpace(NAME, space)
        self.assertEqual(subject.name, NAME)
        self.assertEqual(subject.args, (3.0, 8.0))
        self.assertTrue(subject.func is optuna.trial.Trial.suggest_loguniform)
        
        # discrete uniform
        space = (5.0, 10.0, 2.0)
        subject = opt.SearchableSpace(NAME, space)
        self.assertEqual(subject.name, NAME)
        self.assertEqual(subject.args, (5.0, 10.0, 2.0))
        self.assertTrue(subject.func is 
                        optuna.trial.Trial.suggest_discrete_uniform)
        
        # int
        space = (2, 10)
        subject = opt.SearchableSpace(NAME, space)
        self.assertEqual(subject.name, NAME)
        self.assertEqual(subject.args, (2, 10, 1))
        self.assertTrue(subject.func is optuna.trial.Trial.suggest_int)
        space = (2, 10, 3)
        subject = opt.SearchableSpace(NAME, space)
        self.assertEqual(subject.name, NAME)
        self.assertEqual(subject.args, (2, 10, 3))
        self.assertTrue(subject.func is optuna.trial.Trial.suggest_int)
        
        # catagorical
        space = ['a', 'b', 'c']
        subject = opt.SearchableSpace(NAME, space)
        self.assertEqual(subject.name, NAME)
        self.assertEqual(subject.args, space)
        self.assertTrue(subject.func is optuna.trial.Trial.suggest_catagorical)
        return
