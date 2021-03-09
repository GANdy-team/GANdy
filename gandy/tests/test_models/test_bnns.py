import unittest
import unittest.mock

import tensorflow

import gandy.models.bnns

class TestBNN(unittest.TestCase):
    
    @unittest.mock.patch(
        'gandy.models.bnns.BNN._build')
    def test_prior(self, mocked__build):
        """Ensure handling of bias and kernel size.
        
        Function MUST pass tf.size(kernel), tf.size(bias)
        """
        kernel_size = 5; bias_size = 5
        subject = gandy.models.bnns.BNN((1,), (1,))
        # expected success must return a model
        prior = subject.prior(kernel_size, bias_size)
        self.assertTrue(isinstance(prior, tf.keras.Model))
        # failure cannot parse inputs
        with self.assertRaises(TypeError):
            subject.prior('kernel_size', 'bias_size')
        return
    
    @unittest.mock.patch(
        'gandy.models.bnns.BNN._build')
    def test_posterior(self, mocked__build):
        """Ensure handling of bias and kernel size.
        
        Function MUST pass tf.size(kernel), tf.size(bias)
        """
        kernel_size = 5; bias_size = 5
        subject = gandy.models.bnns.BNN((1,), (1,))
        # expected success must return a model
        prior = subject.posterior(kernel_size, bias_size)
        self.assertTrue(isinstance(prior, tf.keras.Model))
        # failure cannot parse inputs
        with self.assertRaises(TypeError):
            subject.prior('kernel_size', 'bias_size')
        return
    
    @unittest.mock.patch(
        'gandy.models.bnns.BNN._build')
    def test_negative_loglikelihood(self, mocked_build):
        """Input predictions are distributions instead of deterministic.
        
        Distribution should impliment log_prob method
        """
        subject = gandy.models.bnns.BNN((1,), (1,))
        # failure mode, does not have method
        def callable_wo_log_prob():
            return
        with self.assertRaises(TypeError):
            subject.negative_loglikelihood(numpy.array([1,2]),
                                           callable_wo_log_prob)
        # expected success
        mocked_dist = unittest.mock.MagicMock()
        val = subject.negative_loglikelihood('targets',
                                             mocked_dist)
        mocked_dist.log_prob.assert_called_with('targets')
        
        # ability to catch non float
        mocked_dist.return_value = 'string'
        with self.assertRaises(ValueError):
            subject.negative_loglikelihood('targets',
                                            mocked_dist)
        return
    
    def test__build(self):
        """"""
        # start with default initialization
        subject = gandy.models.bnns.BNN((5,), (1,))
        self.assertTrue(isinstance(subject.model, tf.keras.Model))
        self.assertTrue(subject.model._compile_was_called)
        self.assertTrue(subject.model.built)
        self.assertEqual(tuple(subject.model.input.shapes.as_list()),
                         subject.xshape)
        self.assertEqual
        
        