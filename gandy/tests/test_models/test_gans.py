"""Testing functions for UncertaintyModel gan class."""

import unittest
import unittest.mock as mock

import gandy.models.gans as gans
import gandy.models.models

import numpy as np


class TestGAN(unittest.TestCase):
    """Test GAN class."""

    def test_inheritence(self):
        """Ensure the subclass class inherits from parent class."""
        self.assertTrue(issubclass(gans.GAN,
                        gandy.models.models.UncertaintyModel))

    def test__build(self):
        """
        Test build function.

        The build function should create a generator and discriminator.
        This checks both functions are called. It also checks that both
        generator and discriminator are attributes with type == Keras model.
        """
        # CHECK Conditional GAN
        # create gan instance
        subject = gans.GAN(xshape=(10,), yshape=(1,))
        kwargs = dict(noise_shape=(5,))
        subject._build(**kwargs)
        # check attributes
        self.assertTrue(subject.conditional, True)
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
        subject.iterbatches = mock.MagicMock(name='iterbatches',
                                             return_value="Batch1")
        subject._model.fit_gan = mock.MagicMock(name='fit_gan',
                                                return_value='losses')
        kwargs = dict(batches=100)
        losses = subject._train(Xs, Ys, **kwargs)

        # assert fit_gan was called
        subject.iterbatches.assert_called_with(Xs, Ys, 100)
        subject._model.fit_gan.assert_called_with("Batch1")
        self.assertEqual(losses, 'losses')
        return

    @unittest.mock.patch('gandy.models.gans.GAN._build')
    def test__predict(self, mocked__build):
        """
        Test predict function.

        The predict function returns predictions and uncertainties.
        This checks predictions and uncertainties are the appropriate shape
        and the appropriate deepchem calls are made.
        """
        # CHECK Conditional GAN
        Xs = 'Xs'
        subject = gans.GAN(xshape=(4,), yshape=(2, 3))
        subject.conditional = True  # normally set in build
        subject._model.predict_gan_generator = mock.MagicMock(
            name='predict_gan_generator', return_value=np.array([[1]]))
        preds, ucs = subject._predict(Xs)
        subject._model.predict_gan_generator.assert_called_with(
            conditional_inputs=[Xs])
        self.assertEqual(preds, np.array([1]))
        self.assertEqual(ucs, np.array([0]))
        return

    @unittest.mock.patch('gandy.models.gans.GAN._build')
    def test_iterbatches(self, mocked__build):
        """
        Test iterbacthes function.

        The iterbacthes function calls the generate_data function to
        create batches of boostrapped data.
        """
        # check NOT one hot encoded Ys
        Xs = 'Xs'
        Ys = 'Ys'
        subject = gans.GAN(xshape=(4,), yshape=(2,))
        subject._model.batch_size = mock.MagicMock(name='batch_size',
                                                   return_value=100)
        subject.generate_data = mock.MagicMock(
            name='generate_data', return_value=['classes', 'points'])
        kwargs = dict(batches=1)
        result = list(subject.iterbatches(Xs, Ys, **kwargs))
        subject.generate_data.assert_called_with(Xs, Ys,
                                                 subject._model.batch_size)
        data = result[0]
        for key in data.keys():
            self.assertTrue(data[key] in ['classes', 'points'])
        return

    @unittest.mock.patch('gandy.models.gans.GAN._build', return_value='Model')
    def test_generate_data(self, mocked__build):
        """
        Test generate_data function.

        The generate_data function creates batches of boostrapped data.
        """
        Xs = np.array([[1], [2], [3]])
        Ys = np.array([2, 2, 1])
        subject = gans.GAN(xshape=(3,), yshape=(1,))
        classes, points = subject.generate_data(Xs, Ys, 5)
        self.assertEqual(len(classes), 5)
        self.assertEqual(len(points), 5)
        return

    def test_save(self):
        """
        Test save function.

        This checks that a file is written with the appropriate name.
        """
        # test path save
        subject = gans.GAN(xshape=(4,), yshape=(2,))
        subject._model.save = mock.MagicMock(name='save')
        subject.save('path')
        subject._model.save.assert_called_with('path')
        return

    def test_load(self,):
        """
        Test load function.

        This checks that a Keras model instance is returned.
        """
        # test load
        with unittest.mock.patch('tensorflow.keras.models.load_model')\
                as mocked_load:
            subject = gandy.models.gans.GAN.load('filename')
            self.assertTrue(isinstance(subject, gandy.models.gans.GAN))
            mocked_load.assert_called_with('filename')
        return
