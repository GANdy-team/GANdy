"""Unit tests for Metrics module"""
import unittest
import unittest.mock

import numpy as np

import gandy.quality_est.metrics as metrics


class TestMetric(unittest.TestCase):
    """Unit test for Metric parent class"""
    def test___init___(self):
        """Test proper initialization of class with proper inputs"""

        # success case
        subject = metrics.Metric(predictions=np.array([0, 1, 2]),
                                 real=np.array([0, 1, 2]),
                                 uncertainties=np.array([0, 0.5, 1]))

        # check to make sure necessary attributes are inputted
        self.assertTrue(subject.predictions is not None)
        self.assertTrue(subject.real is not None)
        self.assertTrue(subject.calculate() is None)

    def test_calculate(self):
        """Test the calculate function within the parent Metric class"""


class TestMSE(unittest.TestCase):
    """Unit test for MSE subclass"""

    def test_calculate(self):
        """Test the calculate function within the MSE subclass"""

        # failure case: data not iterable
        with self.assertRaises(TypeError):
            subject = metrics.MSE(predictions="0, 1, 2",
                                  real=np.array([0, 1, 2])).calculate()

        with self.assertRaises(TypeError):
            subject = metrics.MSE(predictions=np.array([0, 1, 2]),
                                  real="0, 1, 2").calculate()

        # check to make sure necessary attributes are inputted
        subject = metrics.MSE(predictions=np.array([0, 1, 2]),
                              real=np.array([0, 1, 2]))

        self.assertTrue(subject.predictions is not None)
        self.assertTrue(subject.real is not None)
        self.assertTrue(subject.uncertainties is None)

        # check to make sure output is correct type
        self.assertTrue(isinstance(subject.calculate(), tuple))


class TestRMSE(unittest.TestCase):
    """Unit test for RMSE subclass"""

    def test_calculate(self):
        """Test the calculate function within the RMSE subclass"""

        # failure case: data not iterable
        with self.assertRaises(TypeError):
            subject = metrics.RMSE(predictions="0, 1, 2",
                                   real=np.array([0, 1, 2])).calculate()

        with self.assertRaises(TypeError):
            subject = metrics.RMSE(predictions=np.array([0, 1, 2]),
                                   real="0, 1, 2").calculate()

        # check to make sure necessary attributes are inputted
        subject = metrics.RMSE(predictions=np.array([0, 1, 2]),
                               real=np.array([0, 1, 2]))

        self.assertTrue(subject.predictions is not None)
        self.assertTrue(subject.real is not None)
        self.assertTrue(subject.uncertainties is None)

        # check to make sure output is correct type
        self.assertTrue(isinstance(subject.calculate(), tuple))


class TestF1(unittest.TestCase):
    """Unit test for F1 subclass"""

    def test_calculate(self):
        """Test the calculate function within the F1 subclass"""

        # failure case: data not iterable
        with self.assertRaises(ValueError):
            subject = metrics.F1(predictions="0, 1, 2",
                                 real=np.array([0, 1, 2])).calculate()

        with self.assertRaises(ValueError):
            subject = metrics.F1(predictions=np.array([0, 1, 2]),
                                 real="0, 1, 2").calculate()


        # check to make sure necessary attributes are inputted
        subject = metrics.F1(predictions=np.array([0, 1, 2]),
                             real=np.array([0, 1, 2]))

        self.assertTrue(subject.predictions is not None)
        self.assertTrue(subject.real is not None)
        self.assertTrue(subject.uncertainties is None)

        # check to make sure output is correct type
        self.assertTrue(isinstance(subject.calculate(), tuple))


class TestAccuracy(unittest.TestCase):
    """Unit test for Accuracy subclass"""

    def test_calculate(self):
        """Test the calculate function within the Accuracy subclass"""

        # failure case: data not iterable
        with self.assertRaises(ValueError):
            subject = metrics.Accuracy(predictions="0, 1, 2",
                                       real=np.array([0, 1, 2])).calculate()

        with self.assertRaises(ValueError):
            subject = metrics.Accuracy(predictions=np.array([0, 1, 2]),
                                       real="0, 1, 2").calculate()


        # check to make sure necessary attributes are inputted
        subject = metrics.Accuracy(predictions=np.array([0, 1, 2]),
                                   real=np.array([0, 1, 2]))

        self.assertTrue(subject.predictions is not None)
        self.assertTrue(subject.real is not None)
        self.assertTrue(subject.uncertainties is None)

        # check to make sure output is correct type
        self.assertTrue(isinstance(subject.calculate(), tuple))


class TestUCP(unittest.TestCase):
    """Unit test for UCP subclass"""

    def test_calculate(self):
        """Test the calculate function within the UCP subclass"""

        # failure case: data not iterable
        with self.assertRaises(TypeError):
            subject = metrics.UCP(predictions=np.array([0, 1, 2]),
                                  real="0, 1, 2",
                                  uncertainties=np.array([0, 0.5, 1])).\
                                  calculate()

        with self.assertRaises(TypeError):
            subject = metrics.UCP(predictions="0, 1, 2",
                                  real=np.array([0, 1, 2]),
                                  uncertainties=np.array([0, 0.5, 1])).\
                                  calculate()

        with self.assertRaises(TypeError):
            subject = metrics.UCP(predictions=np.array([0, 1, 2]),
                                  real=np.array([0, 1, 2]),
                                  uncertainties="0, 1, 2").\
                                  calculate()

        # failure case: Uncertainties not given but required
        with self.assertRaises(TypeError):
            subject = metrics.UCP(predictions=np.array([0, 1, 2]),
                                  real=np.array([0, 1, 2])).\
                                  calculate()

        # check to make sure necessary attributes are inputted
        subject = metrics.UCP(predictions=np.array([0, 1, 2]),
                              real=np.array([0, 1, 2]),
                              uncertainties=np.array([0, 0.5, 1]))

        self.assertTrue(subject.predictions is not None)
        self.assertTrue(subject.real is not None)
        self.assertTrue(subject.uncertainties is not None)

        # check to make sure output is correct type
        self.assertTrue(isinstance(subject.calculate(), tuple))
