"""Testing functions for UncertaintyModel gan class."""

# import numpy as np
import unittest
import unittest.mock as mock

# import deepchem

import gandy.models.gans as gans
import gandy.models.models


class TestGAN(unittest.TestCase):
    """Test GAN class."""

    def test_inheritence():
        """Ensure the subclass class inherits from parent class."""
        assert issubclass(gans.gan, gandy.models.models.UncertaintyModel)

    def test__build(self):
        """
        Test build function.

        The build function should create a generator and discriminator.
        This checks both functions are called. It also checks that both
        generator and discriminator are attributes with type == Keras model.
        """
        # CHECK (normal) GAN
        # create gan instance
        subject = gans.GAN(xshape=(4,), yshape=(2,))
        kwargs = dict(noise_shape=(5,))
        subject._build(kwargs)
        # assert create generator function called
        subject.create_generator.assert_called_once_with(kwargs)
        # assert create discriminator function called
        subject.create_discriminator.assert_called_once_with(kwargs)
        # check attributes
        self.assertTrue(hasattr('conditional', False))
        self.assertTrue(hasattr('noise_shape', (5,)))

        # CHECK Conditional GAN
        # create gan instance
        subject = gans.CondGAN(xshape=(4,), yshape=(2, 4))
        kwargs = dict(noise_shape=(5,), n_classes=4)
        subject._build(kwargs)
        # assert create generator function called
        subject.create_generator.assert_called_once_with(kwargs)
        # assert create discriminator function called
        subject.create_discriminator.assert_called_once_with(kwargs)
        # check attributes
        self.assertTrue(hasattr('conditional', True))
        self.assertTrue(hasattr('noise_shape', (5,)))
        self.assertTrue(hasattr('n_classes', 4))
        return

    def test__train(self):
        """
        Test train function.

        The train function calls the fit function for the generator and the
        predict function for the discriminator.
        This checks that there is a Keras callback History object returned.
        """
        Xs = 'Xs'
        Ys = 'Ys'
        subject = gans.GAN(xshape=(4,), yshape=(2,))
        subject.iterbacthes = mock.MagicMock(name='iterbatches',
                                             return_value="Batch1")
        subject.fit_gan = mock.MagicMock(name='fit_gan')
        kwargs = dict(option='x1')
        subject._train(Xs, Ys, kwargs)

        # assert fit_gan was called
        subject.iterbacthes.assert_called_with(Xs, Ys, kwargs)
        subject.fit_gan.assert_called_with("Batch1")
        return

    @unittest.mock.patch('gandy.models.gans.GAN._build', return_value='Model')
    def test__predict(self, mocked__build):
        """
        Test predict function.

        The predict function returns predictions and uncertainties.
        This checks predictions and uncertainties are the appropriate shape
        and the appropriate deepchem calls are made.
        """
        Xs = 'Xs'
        # CHECK (normal) GAN
        subject = gans.gan(xshape=(4,), yshape=(2,))
        subject.predict_gan_generator = mock.MagicMock(
            name='predict_gan_generator', return_value='generated_points')
        preds, ucs = subject._predict(Xs)
        subject._predict.assert_called_with(Xs)
        subject.predict_gan_generator.assert_called_with(None)
        self.assertEqual('preds', 'generated_points')
        self.assertEqual('ucs', None)

        # CHECK Conditional GAN
        Ys = 'Ys'
        subject = gans.gan(xshape=(4,), yshape=(2, 3), n_classes=3)
        subject.predict_gan_generator = mock.MagicMock(
            name='predict_gan_generator', return_value='generated_points')
        with mock.path('deepchem.metrics.to_one_hot',
                       return_value=[10]) as mocked_one_hot:
            preds, ucs = subject._predict(Xs, Ys=Ys)
            mocked_one_hot.assert_called_with(Ys, 3)
            subject._predict.assert_called_with(Xs, Ys=Ys)
            subject.predict_gan_generator.assert_called_with(
                conditional_inputs=[10])
            self.assertEqual('preds', 'generated_points')
            self.assertEqual('ucs', None)
        return

    def test__save(self):
        """
        Test save function.

        This checks that a file is written with the appropriate name.
        """
        return

    def test__load(self):
        """
        Test load function.

        This checks that a Keras model instance is returned.
        """
        return
