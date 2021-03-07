"""Testing functions for UncertaintyModel gan class."""
import numpy as np
import unittest
import unittest.mock

import deepchem

import gandy.models.gans as gans
import gandy.models.models


def TestGAN(unittest.test_case):
	""" Test GAN class."""

    def test_inheritence():
        # ensure the subclass class inherits from both parent classes
        assert issubclass(gans.gan, gandy.models.models.UncertaintyModel)
        assert issubclass(gans.gan, deepchem.models.GAN)

    def test_create_generator(self):
        '''
        Test create generator function.

        The create generator function uses kwargs to create a Keras model.
        This checks that the model compiles.
        '''
        return

    def test_create_discriminator(self):
        '''
        Test create discriminator function.

        The create discriminator function uses kwargs to create a Keras model.
        This checks that the model compiles.
        '''
        return

    def test__build(self):
        '''
        Test build function.

        The build function should create a generator and discriminator.
        This checks both functions are called. It also checks that both
        generator and discriminator are attributes with type == Keras model.
        '''
        # create gan instance
        subject = gans.gan(xshape=(4,), yshape=(2,))
        # create mock functions
        subject.create_generator = mock.MagicMock(name='create_generator')
        subject.create_discriminator = mock.MagicMock(name='create_discriminator')
        kwargs = dict(option=x1)
        subject.build(kwargs)
        # assert create generator function called
        subject.create_generator.assert_called_once_with(kwargs)
        # assert create discriminator function called
        subject.create_discriminator.assert_called_once_with(kwargs)
        # check attributes
        self.assertTrue(hasattr(subject, 'n_classes'))
        self.assertTrue(hasattr(subject, 'generator'))
        self.assertTrue(hasattr(subject, 'discriminator'))
        return

    def test__train(self):
        '''
        Test train function.

        The train function calls the fit function for the generator and the
        predict function for the discriminator.
        This checks that there is a Keras callback History object returned.
        '''
        Xs = 'Xs'
        Ys = 'Ys'
        subject = gans.gan(xshape=(4,), yshape=(2,))
        kwargs = dict(option=x1)
        subject._train(Xs, Ys, kwargs)

        
        # assert generator model called fit and discriminator called predict
        subject.generator.fit.assert_called_with(Xs, Ys, kwargs)
        subject.discriminator.predict.assert_called_with(Xs, Ys, kwargs)
        return

    def test__predict(self):
        '''
        Test predict function.

        The predict function returns predictions and uncertainties.
        This checks predictions and uncertainties are the appropriate shape
        and the appropriate deepchem calls are made.
        '''
        Xs = 'Xs'
        subject = gans.gan(xshape=(4,), yshape=(2,))

        subject.predict_gan_generator = mock.MagicMock(name='predict_gan_generator')

        subject._predict.return_value = ('preds', 'ucs')
        preds, ucs = subject._predict(Xs)
        subject._predict.assert_called_with(Xs)
        self.assertEqual('preds', preds)
        self.assertEqual('ucs', ucs)
        return

    def test__save(self):
        '''
        Test save function.

        This checks that a file is written with the appropriate name.
        '''
        return

    def test__load(self):
        '''
        Test load function.

        This checks that a Keras model instance is returned.
        '''
        return