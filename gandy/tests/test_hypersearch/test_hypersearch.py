"""Testing hyperparameter optimization with optuna"""
import itertools
import unittest
import unittest.mock

import numpy
import optuna.trial

import gandy.optimization.optimization as opt
import gandy.models.models

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

    
class TestSubjectObjective(unittest.TestCase):
    
    params = [opt.SearchableSpace('hyp1', (1,10)),
              opt.SearchableSpace('hyp2', ['a', 'b'])]
    inputs = {'subject': gandy.models.models.UncertaintyModel,
              'Xs': numpy.array(1),
              'Ys': numpy.array(1),
              'param_space': params,
              } 
    def test___init__(self):
        """Ensure only one validation option and proper saving of parameters"""
        # expected success, no sessions specified, no val
        subject = opt.SubjectObjective(**self.inputs)
        self.assertTrue(subject.sessions is range(1))
        self.assertTrue(subject.param_space is params)
        for att in ['k', 'val_data', 'val_frac']:
            self.assertEqual(getattr(subject, att), None)
        # specify sessions
        subject = opt.SubjectObjective(**self.inputs, sessions=5)
        self.assertTrue(subject.sessions is range(5))
        subject = opt.SubjectObjective(**self.inputs, sessions=['a', 'b'])
        self.assertEqual(subject.sessions, ['a', 'b'])
        # test proper validation handling
        # k
        subject = opt.SubjectObjective(**self.inputs, k=5)
        self.assertTrue(subject.k is not None)
        subject = opt.SubjectObjective(**self.inputs, 
                                       k=[(numpy.array(1),numpy.array(1))])
        self.assertTrue(subject.k is not None)
        # val_data
        subject = opt.SubjectObjective(**self.inputs, 
                                       val_data=(numpy.array(1),
                                                 numpy.array(1)))
        self.assertTrue(subject.val_data is not None)
        # val_frac
        subject = opt.SubjectObjective(**self.inputs, 
                                       val_frac=0.5)
        self.assertTrue(subject.val_frac is not None)
        # test failure cases - cannot have two of these options
        failure_cases = itertools.combinations(
            ['k', 'val_data', 'val_frac'], 2
        )
        for fc in failure_cases:
            kws = dict(zip(fc, ['keywordvalue1', 'keywordvalue2']))
            with self.assertRaises('ValueError'):
                subject = opt.SubjectObjective(**self.inputs, **kws)
                
        # ensure proper saving of keyword arguments
        subject = opt.SubjectObjective(**self.inputs, keyword1=5)
        self.assertTrue('keyword1' in subject.kwargs.keys())
        
        # kwarg and param overlap - would cause issues later
        with self.assertRaises(ValueError):
            subject = opt.SubjectObjective(**self.inputs, hyp1=5)
        return
    
    @unittest.mock.patch('sklearn.model_selection.KFold')
    def test_property_k(self, mocked_kfold):
        """ensure proper handling of kfolds setting"""
        mocked_kfold.split.return_value = ('train', 'test')
        subject = opt.SubjectObjective(**self.inputs)
        # int or iterable of tuple works
        subject.k = 5
        mocked_kfold.assert_called()
        mocked_kfold.split.called_with(subject.Xs)
        self.assertTrue(mocked_kfold.split.call_count == 5)
        self.assertTrue(type(subject.k) == list)
        for f in subject.k:
            self.assertTrue(type(f) == tuple and len(f) == 2)
        test_folds = [(1,2), (1,2)]
        subject.k = test_folds
        self.assertEqual(test_folds, subject.k)
        # failure case not either
        with self.assertRaises(TypeError):
            subject.k = 'str'
        return
    
    def test_property_val_data(self):
        """ability to check val data before saving"""
        val_data_failures = [(1, 3), 
                             [numpy.array(1), numpy.array(1)],
                             (numpy.array([1,2]), numpy.array(1))]
        subject = opt.SubjectObjective(**self.inputs)
        for val_data in val_data_failures:
            with self.assertRaises(TypeError):
                subject.val_data = val_data
                
        # success - tuple of arrays of the same length
        val_data = (numpy.array([1,2,3]), numpy.array([1,2,3]))
        subject.val_data = val_data
        self.assertTrue(subject.val_data is not None)
        return
    
    def test_property_val_frac(self):
        """it should only accept float between 0 and 1"""
        subject = opt.SubjectObjective(**self.inputs)
        val_frac_failures = [0.0, 1.0, 2.0]
        for val_frac in val_frac_failures:
            with self.assertRaises(ValueError):
                subject.val_frac = val_frac
        with self.assertRaises(TypeError):
            subject.val_frac = 'string'
        return
    
    @unittest.mock.patch('optuna.trial.Trial'):
    def test__sample_params(self, mocked_Trial):
        """can the sampler get parameters from optuna methods"""
        # prepare the mocked trial
        trial = mocked_Trial()
        subject = opt.SubjectObjective(**self.inputs)
        # run the sample and test correct calls
        params = subject._sample_params(trial)
        mocked_Trial.suggest_int.assert_called_with(
            trial, 'hyp1', 1, 10, 1
        )
        mocked_Trial.suggest_catagorical.assert_called_with(
            trial, 'hyp2', ['a', 'b']
        )
        self.assertTrue(all(hyp in params.keys() for hyp in ['hyp1', 'hyp2']))
        return
    
    @unittest.mock.patch('import gandy.models.models.UncertaintyModel')
    def test__execute_instance(self, mocked_UM):
        """does the method instantialize and call the correct model methods"""
        subject = opt.SubjectObjective(**self.inputs, xshape=(5,), keyword=5)
        hyparams = {'hp1': 1, 'hp2': 2}
        train_data = ('Xst', 'Yst')
        val_data = ('Xsv', 'Ysv')
        mocked_UM_in = unittest.mock.MagicMock() 
        mocked_UM.return_value = mocked_UM_in
        mocked_UM_in.score.return_value = 'score'
        # execute the instance
        score = subject._execute_instance(hyparams, train_data, val_data)
        mocked_UM.assert_called_with(xshape=(5,), 
                                     keyword=5,
                                     hp1=1,
                                     hp2=2)
        mocked_UM_in.train.assert_called_with('Xst', 'Yst',
                                              xshape=(5,), 
                                              keyword=5,
                                              hp1=1,
                                              hp2=2)
        mocked_UM_in.score.assert_called_with('Xsv', 'Ysv',
                                              xshape=(5,), 
                                              keyword=5,
                                              hp1=1,
                                              hp2=2)
        self.assertTrue(score is 'score')
        return
    
    @unittest.mock.patch('optuna.trial.Trial')
    def test___call__(self, mocked_Trial):
        """ability to identify different validation options and call the 
        correct methods"""
        subject = opt.SubjectObjective(**self.inputs)
        mocked_sample = unittest.mock.MagicMock(
            return_value={'hp1': 1, 'hp2': 2}
        )
        subject._sample_params = mocked_sample
        mocked_execute = unittest.mock.MagicMock(
            return_value='loss'
        )
        subject._execute_instance = mocked_execute
        trial = mocked_Trial()
        
        # start with k specifed - folds are arrays of indexes data len = 1
        first, second = (np.array(0), np.array(0)), (np.array(0), np.array(0))
        subject._k = [first, second]
        subject.__call__(trial)
        mocked_sample.assert_called_with(trial)
        mocked_execute.assert_called_with(
            {'hp1': 1, 'hp2': 2}, 
            (subject.Xs[second[0]], subject.Ys[second[0]]), 
            (subject.Xs[second[0]], subject.Ys[second[0]])
        )
        self.assertTrue(mocked_execute.call_count == 2)
        trial.should_prune.assert_called()
        # reset calls
        mocked_execute.reset_mock()
        mocked_sample.reset_mock()
        trial.reset_mock()
        
        # val_data specifed
        val_data = ('Xsv', 'Ysv')
        subject._val_data = val_data
        subject.__call__(trial)
        mocked_sample.assert_called_with(trial)
        mocked_execute.assert_called_with(
            {'hp1': 1, 'hp2': 2}, 
            (subject.Xs, subject.Ys), 
            val_data
        )
        self.assertTrue(mocked_execute.call_count == 1)
        trial.should_prune.assert_called()
        # reset calls
        mocked_execute.reset_mock()
        mocked_sample.reset_mock()
        trial.reset_mock()
        
        # val frac specified
        val_frac = 5.0
        subject._val_frac = val_frac
        
        with unittest.mock.patch('sklearn.model_selection.train_test_split',
                                 return_value = 'Xt', 'Xv', 'Yt', 'Yv') as
                                 mocked_tts:
            subject.__call__(trial)
            mocked_sample.assert_called_with(trial)
            mocked_tts.assert_called_with(subject.Xs, subject.Ys, 
                                          test_size=val_frac)
            mocked_execute.assert_called_with(
                {'hp1': 1, 'hp2': 2}, 
                ('Xt', 'Yt'), 
                ('Xv', 'Yv')
            )
            self.assertTrue(mocked_execute.call_count == 1)
            trial.should_prune.assert_called()
        # reset calls
        mocked_execute.reset_mock()
        mocked_sample.reset_mock()
        trial.reset_mock()
        subject._val_frac = None
            
        # None specified
        with unittest.mock.patch('sklearn.model_selection.train_test_split',
                                 return_value = 'Xt', 'Xv', 'Yt', 'Yv') as
                                 mocked_tts:
            subject.__call__(trial)
            mocked_sample.assert_called_with(trial)
            mocked_tts.assert_called_with(subject.Xs, subject.Ys, 
                                          test_size=0.8)
            mocked_execute.assert_called_with(
                {'hp1': 1, 'hp2': 2}, 
                ('Xt', 'Yt'), 
                ('Xv', 'Yv')
            )
            self.assertTrue(mocked_execute.call_count == 1)
            trial.should_prune.assert_called()
        return
    
class TestOptRoutine(unittest.TestCase):
    """User interface class"""
    
    def test___init__(self):
        """proper saving of keyword arguments and data saving"""
        # failure case not correct model type
        with self.assertRaises(TypeError):
            subject = opt.OptRoutine(subject=opt.SearchableSpace,
                                     Xs=numpy.array([1,2,3]),
                                     Ys=numpy.array([1,2,3]),
                                     search_space={'hyp1': (1,10),
                                                     'hyp2': ['a', 'b']},
                                     keyword=5)
        # failure case data not iterable
        with self.assertRaises(TypeError):
            subject = opt.OptRoutine(subject = gandy.models.models.\
                                     UncertaintyModel,
                                     Xs='str',
                                     Ys=numpy.array([1,2,3]),
                                     search_space = {'hyp1': (1,10),
                                                     'hyp2': ['a', 'b']},
                                     keyword=5)
        with self.assertRaises(TypeError):
            subject = opt.OptRoutine(subject=gandy.models.models.\
                                     UncertaintyModel,
                                     Xs=numpy.array([1,2,3]),
                                     Ys='str',
                                     search_space={'hyp1': (1,10),
                                                   'hyp2': ['a', 'b']},
                                     keyword=5)
        
        