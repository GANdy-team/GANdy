import unittest
import unittest.mock

import numpy
import tensorflow as tf

import gandy.models.bnns


class TestBNN(unittest.TestCase):

    @unittest.mock.patch(
        'gandy.models.bnns.BNN._build')
    def test_prior(self, mocked__build):
        """Ensure handling of bias and kernel size.

        Function MUST pass tf.size(kernel), tf.size(bias)
        """
        kernel_size = 5
        bias_size = 5
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
        kernel_size = 5
        bias_size = 5
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
            subject.negative_loglikelihood(numpy.array([1, 2]),
                                           callable_wo_log_prob)
        # expected success
        mocked_dist = unittest.mock.MagicMock()
        subject.negative_loglikelihood('targets',
                                       mocked_dist)
        mocked_dist.log_prob.assert_called_with('targets')

        # ability to catch non float
        mocked_dist.return_value = 'string'
        with self.assertRaises(ValueError):
            subject.negative_loglikelihood('targets',
                                           mocked_dist)
        return

    def test__build(self):
        """The build should pass kwargs to the correct place.

        We need to ensure the returned keras model is both compiled
        and built.
        """
        # start with default initialization
        subject = gandy.models.bnns.BNN((5,), (1,))
        self.assertTrue(isinstance(subject.model, tf.keras.Model))
        self.assertTrue(subject.model._compile_was_called)
        self.assertTrue(subject.model.built)
        self.assertEqual(tuple(subject.model.input.shapes.as_list()),
                         subject.xshape)
        self.assertEqual(tuple(subject.model.output.shapes.as_list()),
                         subject.yshape)

        # test keyword assignment
        with unittest.mock.patch(
            'tensorflow.keras.Sequential.compile'
        ) as mocked_compile:
            subject = gandy.models.bnns.BNN((5,), (1,),
                                            optimizer='rms_prop',
                                            metrics=['MSE'])
            mocked_compile.assert_called_with(optimizer='rms_prop',
                                              metrics=['MSE'])
        return

    def test__train(self):
        """We just want to call the host fit method"""
        Xs = 'Xs'
        Ys = 'Ys'
        with unittest.mock.patch(
            'tensorflow.keras.Sequential.fit'
        ) as mocked_fit:
            subject = gandy.models.bnns.BNN((5,), (1,))
            subject._train(Xs, Ys, epochs=10)
            mocked_fit.assert_called_with(Xs, Ys, epochs=10)
            return

    def test__predict(self):
        """Predict for a probabilistic BNN is just letting the tensors
        flow, make sure it is passed to input.
        """
        subject = gandy.models.bnns.BNN((5,), (1,))
        subject.model = unittest.mock.MagicMock()
        subject._predict('Xs')
        subject.model.assert_called_with('Xs')
        return

    def test_save(self):
        """Save should just call keras save"""
        with unittest.mock.patch(
            'tensorflow.keras.Sequential.save'
        ) as mocked_save:
            subject = gandy.models.bnns.BNN((5,), (1,))
            subject.save('filename')
            mocked_save.assert_called_with('filename')
        return

    def test_load(self):
        """load needs to use keras load, but then also stick it into a gandy
        model with the correct shape
        """
        model_mocked = unittest.mock.MagicMock()
        model_mocked.input.shape.to_list.return_value = [5, ]
        model_mocked.output.shape.to_list.return_value = [3, ]
        with unittest.mock.patch(
            'tensorflow.keras.models.load',
            return_value=model_mocked
        ) as mocked_load:
            subject = gandy.models.bnns.BNN.load('filename')
            self.assertTrue(isinstance(subject), gandy.models.bnns.BNN)
            self.assertEqual(subject.xhsape, (5,))
            self.assertEqual(subject.xhsape, (3,))
            mocked_load.assert_called_with('filename')
        return
