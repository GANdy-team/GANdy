import unittest
import unittest.mock

import numpy
import tensorflow as tf
import tensorflow_probability as tfp

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
        subject = gandy.models.bnns.BNN((1,), (1,), train_size=5)
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
        subject = gandy.models.bnns.BNN((1,), (1,), train_size=5)
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
        subject = gandy.models.bnns.BNN((1,), (1,), train_size=5)
        # failure mode, does not have method

        def callable_wo_log_prob():
            return
        with self.assertRaises(AttributeError):
            subject.negative_loglikelihood(numpy.array([1, 2]),
                                           callable_wo_log_prob)
        # expected success
        mocked_dist = unittest.mock.MagicMock()
        mocked_dist.log_prob.return_value = 5.0
        subject.negative_loglikelihood('targets',
                                       mocked_dist)
        mocked_dist.log_prob.assert_called_with('targets')

        return

    def test__build(self):
        """The build should pass kwargs to the correct place.

        We need to ensure the returned keras model is both compiled
        and built.
        """
        x = numpy.array([[1, 2],
                         [3, 4],
                         [5, 6]])
        # start with default initialization
        subject = gandy.models.bnns.BNN((2,), (4,), train_size=len(x))
        self.assertTrue(isinstance(subject.model, tf.keras.Model))
        self.assertTrue(subject.model._compile_was_called)
        self.assertTrue(subject.model.built)
        self.assertEqual(tuple(subject.model.input.shape.as_list())[1:],
                         subject.xshape)
        predict = subject.model.predict(x)
        self.assertTrue(predict.shape == (3, 4))
        out = subject.model(x)
        self.assertTrue(isinstance(out, tfp.distributions.Distribution))

        # test keyword assignment
        subject = gandy.models.bnns.BNN((2,), (4,),
                                        train_size=len(x),
                                        optimizer='RMSprop')
        self.assertTrue(isinstance(subject.model.optimizer,
                                   tf.keras.optimizers.RMSprop))
        subject = gandy.models.bnns.BNN((2,), (4,),
                                        train_size=len(x),
                                        optimizer=tf.keras.optimizers.RMSprop)
        self.assertTrue(isinstance(subject.model.optimizer,
                                   tf.keras.optimizers.RMSprop))
        opt = tf.keras.optimizers.RMSprop()
        subject = gandy.models.bnns.BNN(
            (2,), (4,),
            train_size=len(x),
            optimizer=opt
        )
        self.assertTrue(subject.model.optimizer is opt)
        return

    def test__train(self):
        """We just want to call the host fit method"""
        Xs = 'Xs'
        Ys = 'Ys'
        mocked_fit = unittest.mock.MagicMock(return_value='loss')
        subject = gandy.models.bnns.BNN((5,), (1,), train_size=2)
        subject.model.fit = mocked_fit
        losses = subject._train(Xs, Ys, epochs=10)
        mocked_fit.assert_called()
        self.assertEqual(losses, 'loss')
        return

    def test__predict(self):
        """Predict for a probabilistic BNN is just letting the tensors
        flow, make sure it is passed to input.
        """
        subject = gandy.models.bnns.BNN((5,), (1,), train_size=2)
        dists = unittest.mock.MagicMock()
        subject._model = unittest.mock.MagicMock(return_value=dists)
        subject._predict('Xs')
        subject.model.assert_called_with('Xs')
        dists.mean.assert_called()
        dists.stddev.assert_called()
        return

    def test_save(self):
        """Save should just call keras save"""
        mocked_save = unittest.mock.MagicMock()
        subject = gandy.models.bnns.BNN((5,), (1,), train_size=2)
        subject.model.save = mocked_save
        subject.save('filename')
        mocked_save.assert_called_with('filename')
        return

    def test_load(self):
        """load needs to use keras load, but then also stick it into a gandy
        model with the correct shape
        """
        with unittest.mock.patch(
            'tensorflow.keras.models.load_model'
        ) as mocked_load:
            subject = gandy.models.bnns.BNN.load('filename')
            self.assertTrue(isinstance(subject, gandy.models.bnns.BNN))
            mocked_load.assert_called_with('filename')
        return
