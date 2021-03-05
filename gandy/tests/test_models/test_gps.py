"""Testing functions for UncertaintyModel gaussian process class"""
import numpy
import unittest
import unittest.mock

import sklearn.gaussian_process

import gandy.models.gps as gps
import gandy.models.models

## ensure the class inherits
assert issubclass(gps.ucGaussianProcess, gandy.models.models.UncertaintyModel)

def TestGaussianProcess(unittest.test_case):
    
    @unittest.mock.patch('sklearn.gaussian_process')
    def test__build(self, mocked_gp):
        """Ensure the child method creates sklearn GP"""
        # set up mocks
        mocked_gp.GaussianProcessRegressor.return_value = 'Regressor'
        mocked_gp.GaussianProcessClassifier.return_value = 'Classifer'
        # run both options and test calls
        # init and build methods already tested in parent
        # we know init kwargs get to here
        with self.assertRaises(ValueError):
            subject = gps.ucGaussianProcess((1,), (1,),
                                            model_type = 'something')
        subject = gps.ucGaussianProcess((1,), (1,),
                                        model_type = 'classifer',
                                        keyword=5)
        mocked_gp.GaussianProcessClassifier.called_with(keyword=5)
        self.assertEqual(subject.model, 'Classifer')
        subject = gps.ucGaussianProcess((1,), (1,),
                                        model_type = 'regressor',
                                        keyword=5)
        mocked_gp.GaussianProcessRegressor.called_with(keyword=5)
        self.assertEqual(subject.model, 'Regressor')
        return
    
    @unittest.mock.patch('sklearn.gaussian_process')
    def test__train(self, mocked_gp):
        """Ensure the model's fit method is called"""
        Xs = 'Xs'; Ys = 'Ys'
        subject = gps.ucGaussianProcess((1,), (1,),
                                        model_type = 'classifer')
        subject._train(Xs, Ys, keyword=5)
        subject.model.fit.assert_called_with(Xs, Ys, keyword=5)
        subject = gps.ucGaussianProcess((1,), (1,),
                                        model_type = 'regressor')
        subject._train(Xs, Ys, keyword=5)
        subject.model.fit.assert_called_with(Xs, Ys, keyword=5)
        return
    
    @unittest.mock.patch('sklearn.gaussian_process')
    def test__predict(self):
        """Ensure the proper calls with return_std keyword"""
        Xs = 'Xs'
        # classifer
        subject = gps.ucGaussianProcess((1,), (1,),
                                        model_type = 'classifer')
        subject.model.predict.return_value = 'preds'
        subject.model.predict_proba.return_value = 'uncs'
        # execute the method
        preds, uncs = subject._predict(Xs)
        subject.model.predict.assert_called_with(Xs)
        subject.model.predict_proba.assert_called_with(Xs)
        self.assertEqual('preds', preds)
        self.assertEqual('uncs', uncs)
        # regressor
        subject = gps.ucGaussianProcess((1,), (1,),
                                        model_type = 'regressor')
        subject.model.predict.return_value = ('preds', 'uncs')
        # execute the method
        preds, uncs = subject._predict(Xs)
        subject.model.predict.assert_called_with(Xs, return_std=True)
        self.assertEqual('preds', preds)
        self.assertEqual('uncs', uncs)
        return
    
    def test_R(self):
        """test direct regressor instantialization"""
        subject = gps.ucGaussianProcess.R((1,), (1,))
        self.assertTrue(
            isinstance(subject.model, 
                       sklearn.gaussian_process.GaussianProcessRegressor)
        )
        return
    
    def test_C(self):
        """test direct regressor instantialization"""
        subject = gps.ucGaussianProcess.C((1,), (1,))
        self.assertTrue(
            isinstance(subject.model, 
                       sklearn.gaussian_process.GaussianProcessClassifier)
        )
        return
    