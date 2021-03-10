"""Unit tests for Metrics module"""

import unittest
import unittest.mock

import numpy as np

import gandy.quality_est.metrics as metrics


class TestMetric(unittest.TestCase):
    """Unit test for Metric parent class"""

    def test___init___(self):
        """Test proper initialization of class with proper inputs"""

        # failure cases: data not iterable
        with self.assertRaises(TypeError):
            subject = metrics.Metric(predictions="0, 1, 2",
                                     real=np.array([0, 1, 2]),
                                     uncertainties=np.array([0, 0.5, 1]))

        with self.assertRaises(TypeError):
            subject = metrics.Metric(predictions=np.array([0, 1, 2]),
                                     real="0, 1, 2",
                                     uncertainties=np.array([0, 0.5, 1]))

        with self.assertRaises(TypeError):
            subject = metrics.Metric(predictions=np.array([0, 1, 2]),
                                     real=np.array([0, 1, 2]),
                                     uncertainties="0, 1, 2")

        with self.assertRaises(TypeError):
            subject = metrics.Metric(predictions=np.array([0, 1, 2]),
                                     real="0, 1, 2")

        # success case
        subject = metrics.Metric(predictions=np.array([0, 1, 2]),
                                 real=np.array([0, 1, 2]),
                                 uncertainties=np.array([0, 0.5, 1]))

        # check to make sure necessary attributes are inputted
        self.assertTrue(subject.predictions is not None)
        self.assertTrue(subject.real is not None)

    def test_calculate(self):
        """Test the calculate function within the parent Metric class"""

        # ensure calculate method is called using mock function
        subject = metrics.Metric
        subject.calculate = unittest.mock.MagicMock(name='calculate')
        subject.calculate.assert_called_once_with(kwargs)


class TestMSE(unittest.TestCase):
    """Unit test for MSE subclass"""

    def test_calculate(self):
        """Test the calculate function within the parent Metric class"""

        # failure case: data not iterable
        with self.assertRaises(TypeError):
            subject = metric.MSE(predictions="0, 1, 2",
                                 real=np.array([0, 1, 2]))

        with self.assertRaises(TypeError):
            subject = metric.MSE(predictions=np.array([0, 1, 2]),
                                 real="0, 1, 2")

        # failure case: uncertainties given when None expected
        with self.assertRaises(TypeError):
            subject = metric.MSE(predictions=np.array([0, 1, 2]),
                                 real=np.array([0, 1, 2]),
                                 uncertainties=np.array([0, 1, 2]))

        # check to make sure necessary attributes are inputted
        subject = metrics.MSE(predictions=np.array([0, 1, 2]),
                              real=np.array([0, 1, 2]))

        self.assertTrue(subject.predictions is not None)
        self.assertTrue(subject.real is not None)
        self.assertTrue(subject.uncertainties is None)

        # check to make sure output is correct type
        self.assertTrue(isinstance(subject, tuple))


class TestRMSE(unittest.TestCase):
    """Unit test for RMSE subclass"""

    def test_calculate(self):
        """Test the calculate function within the parent Metric class"""

        # failure case: data not iterable
        with self.assertRaises(TypeError):
            subject = metric.RMSE(predictions="0, 1, 2",
                                  real=np.array([0, 1, 2]))

        with self.assertRaises(TypeError):
            subject = metric.RMSE(predictions=np.array([0, 1, 2]),
                                  real="0, 1, 2")

        # failure case: uncertainties given when None expected
         with self.assertRaises(TypeError):
            subject = metric.RMSE(predictions=np.array([0, 1, 2]),
                                  real=np.array([0, 1, 2]),
                                  uncertainties=np.array([0, 1, 2]))

        # check to make sure necessary attributes are inputted
        subject = metrics.RMSE(predictions=np.array([0, 1, 2]),
                               real=np.array([0, 1, 2]))

        self.assertTrue(subject.predictions is not None)
        self.assertTrue(subject.real is not None)
        self.assertTrue(subject.uncertainties is None)

        # check to make sure output is correct type
        self.assertTrue(isinstance(subject, tuple))


class TestF1(unittest.TestCase):
    """Unit test for F1 subclass"""

    def test_calculate(self):
        """Test the calculate function within the parent Metric class"""

        # failure case: data not iterable
        with self.assertRaises(TypeError):
            subject = metric.F1(predictions="0, 1, 2",
                                real=np.array([0, 1, 2]))

        with self.assertRaises(TypeError):
            subject = metric.F1(predictions=np.array([0, 1, 2]),
                                real="0, 1, 2")
    
        # failure case: uncertainties given when None expected 
        with self.assertRaises(TypeError):
            subject = metric.F1(predictions=np.array([0, 1, 2]),
                                real=np.array([0, 1, 2]),
                                uncertainties=np.array([0, 1, 2]))

       # check to make sure necessary attributes are inputted
        subject = metrics.F1(predictions=np.array([0, 1, 2]),
                             real=np.array([0, 1, 2]))
    
        self.assertTrue(subject.predictions is not None)
        self.assertTrue(subject.real is not None)
        self.assertTrue(subject.uncertainties is None)

        # check to make sure output is correct type
        self.assertTrue(isinstance(subject, (float, int)))
