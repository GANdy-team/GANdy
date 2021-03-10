"""Testing functions for UncertaintyModel gan class."""

import unittest
import unittest.mock as mock

import gandy.models.gans as gans
import gandy.models.models


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
        # CHECK (normal) GAN
        # create gan instance
        subject = gans.GAN(xshape=(4,), yshape=(2,))
        kwargs = dict(noise_shape=(5,))
        subject._build(**kwargs)
        # created mocked functions
        subject._model.create_generator = unittest.mock.MagicMock(
            name='create_generator')
        subject._model.create_discriminator = unittest.mock.MagicMock(
            name='create_discriminator')
        # assert create generator function called
        subject._model.create_generator.assert_called_once_with(kwargs)
        # assert create discriminator function called
        subject._model.create_discriminator.assert_called_once_with(kwargs)
        # check attributes
        self.assertTrue(subject.conditional, False)
        self.assertTrue(subject.noise_shape, (5,))

        # CHECK Conditional GAN
        # create gan instance
        subject = gans.CondGAN(xshape=(4,), yshape=(2, 4))
        self.assertTrue(issubclass(gans.CondGAN, gans.GAN))
        kwargs = dict(noise_shape=(5,), n_classes=4)
        subject._build(**kwargs)
        # assert create generator function called
        subject._model.create_generator.assert_called_once_with(kwargs)
        # assert create discriminator function called
        subject._model.create_discriminator.assert_called_once_with(kwargs)
        # check attributes
        self.assertTrue(subject.conditional, True)
        self.assertTrue(subject.noise_shape, (5,))
        self.assertTrue(subject.n_classes, 4)
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
        subject._model.fit_gan = mock.MagicMock(name='fit_gan')
        kwargs = dict(option='x1')
        subject._train(Xs, Ys, **kwargs)

        # assert fit_gan was called
        subject.iterbacthes.assert_called_with(Xs, Ys, **kwargs)
        subject._model.fit_gan.assert_called_with("Batch1")
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
        subject = gans.GAN(xshape=(4,), yshape=(2,))
        subject.predict_gan_generator = mock.MagicMock(
            name='predict_gan_generator', return_value='generated_points')
        preds, ucs = subject._predict(Xs)
        subject._model.predict_gan_generator.assert_called_with(None)
        self.assertEqual(preds, 'generated_points')
        self.assertEqual(ucs, None)

        # CHECK Conditional GAN
        Ys = 'Ys'
        subject = gans.GAN(xshape=(4,), yshape=(2, 3), n_classes=3)
        subject._model.predict_gan_generator = mock.MagicMock(
            name='predict_gan_generator', return_value='generated_points')
        with mock.patch('deepchem.metrics.to_one_hot',
                        return_value=[10]) as mocked_one_hot:
            preds, ucs = subject._predict(Xs, Ys=Ys)
            mocked_one_hot.assert_called_with(Ys, 3)
            subject._model.predict_gan_generator.assert_called_with(
                conditional_inputs=[10])
            self.assertEqual(preds, 'generated_points')
            self.assertEqual(ucs, None)
        return

    @unittest.mock.patch('gandy.models.gans.GAN._build', return_value='Model')
    def test_iterbacthes(self, mocked__build):
        """
        Test iterbacthes function.

        The iterbacthes function calls the generate_data function to
        create batches of boostrapped data.
        """
        # check NOT one hot encoded Ys
        Xs = 'Xs'
        Ys = 'Ys'
        subject = gans.GAN(xshape=(4,), yshape=(2,), n_classes=10)
        subject.generate_data = mock.MagicMock(
            name='generate_data', return_value=('classes', 'points'))
        kwargs = dict(bacthes=1, batch_size=5)
        with mock.patch('deepchem.metrics.to_one_hot',
                        return_value='one_hot_classes') as mocked_one_hot:
            result = list(subject.iterbatches(Xs, Ys, **kwargs))
            subject.generate_data.assert_called_with(Xs, Ys, 5)
            expected_result = {subject._model.data_inputs[0]: 'points',
                               subject._model.conditional_inputs[0]:
                               'classes'}
            self.assertEqual(expected_result, result)
        # check one hot encoded Ys
        Xs = 'Xs'
        Ys = [[0, 1], [1, 0], [1, 0]]
        subject = gans.GAN(xshape=(4,), yshape=(2,), n_classes=10)
        subject.generate_data = mock.MagicMock(
            name='generate_data', return_value=('classes', 'points'))
        kwargs = dict(bacthes=1, batch_size=5)
        with mock.patch('deepchem.metrics.to_one_hot',
                        return_value='one_hot_classes') as mocked_one_hot:
            result = list(subject.iterbacthes(Xs, Ys, **kwargs))
            subject.generate_data.assert_called_with(Xs, Ys, 5)
            mocked_one_hot.assert_called_with('classes', 10)
            expected_result = {subject._model.data_inputs[0]: 'points',
                               subject._model.conditional_inputs[0]:
                               'one_hot_classes'}
            self.assertEqual(expected_result, result)
        return

    @unittest.mock.patch('gandy.models.gans.GAN._build', return_value='Model')
    def test_generate_data(self, mocked__build):
        """
        Test generate_data function.

        The generate_data function creates batches of boostrapped data.
        """
        Xs = ['x1', 'x2', 'x3']
        Ys = ['y1', 'y2', 'y3']
        subject = gans.GAN(xshape=(4,), yshape=(2,), n_classes=1)
        classes, points = subject.generate_data(Xs, Ys, 5)
        self.assertEqual(len(classes), 5)
        self.assertEqual(len(points), 5)
        return

    @unittest.mock.patch('gandy.models.gans.GAN._build', return_value='Model')
    def test_save(self, mocked__build):
        """
        Test save function.

        This checks that a file is written with the appropriate name.
        """
        # test path save
        subject = gans.GAN(xshape=(4,), yshape=(2,), n_classes=10)
        subject._model.save = mock.MagicMock(name='save')
        subject.save('path')
        subject._model.save.assert_called_with('path')
        # test h5 save
        subject.save('test_model.h5')
        subject._model.save.assert_called_with('test_model.h5')
        return

    @unittest.mock.patch('tf.keras.models.load_model', return_value='Model')
    def test_load(self, mocked_load):
        """
        Test load function.

        This checks that a Keras model instance is returned.
        """
        # test load
        subject = gans.GAN.load('test_model.h5')
        self.assertEqaul(subject, 'Model')
        return
