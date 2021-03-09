"""
This class implements a GAN using deepchem's GAN class.

Deepchem's tutorial on GANs (14_Conditional_Generative_Adversarial_Networks)
can be found here:
https://github.com/deepchem/deepchem/blob/master/examples/tutorials/
    14_Conditional_Generative_Adversarial_Networks.ipynb

See dcgan for the implemented deepchem GAN and conditional GAN.
"""

# gandy imports
import gandy.models.models
import gandy.metrics

# deep learning imports
import deepchem
import gandy.models.dcgan as dcgan
import tensorflow as tf

# typing imports
from typing import Any, Object, Type, Callable

# typing
import numpy as np
Array = Type[np.ndarray]


class GAN(gandy.models.models.UncertaintyModel):
    """
    Implements Generative Adversarial Networks.

    A Generative Adversarial Network (GAN) is a type of generative model.
    It consists of two parts called the "generator" and the "discriminator".
    The generator takes random noise as input and transforms it into an
    output that (hopefully) resembles the training data. The discriminator
    takes a set of samples as input and tries to distinguish the real
    training samples from the ones created by the generator. Both of them
    are trained together. The discriminator tries to get better and better
    at telling real from false data, while the generator tries to get better
    and better at fooling the discriminator.

    Thank you to deepchem at
    https://github.com/deepchem/deepchem/blob/master/deepchem/models/gan.py#L14-L442
    for the information about GANS.
    """

    # overridden method from UncertaintyModel class
    def _build(self, **kwargs) -> Object:
        """
        Construct the model.

        This instantiates the deepchem gan as the model.

        Arguments:
            **kwargs - key word arguments for creating the generator
                and discriminator. See dcgan.create_generator and
                dcgan.create_discriminator for those kwargs.
                type == dict

        Returns:
            model - Deepchem GAN model found in dcgan
                type == Object
        """
        # get noise shape from kwargs
        # default is 10 dimensional
        noise_shape = kwargs.get('noise_shape', (10,))
        # get n_classes from kwargs, default None
        n_classes = kwargs.get('n_classes', None)

        # determine whether to use gan or conditional gan
        if n_classes is not None:
            # if number of classes is specified, assumes conditional GAN
            self.conditional = True
            # Should this be flagged somewhere?...
            if self.yshape[0] > 1:
                # Ys are already one hot encoded
                n_classes = kwargs.get('n_classes', self.yshape[0])
            else:
                # Ys are NOT one hot encoded
                # Or this is regression, which would be == 1
                n_classes = kwargs.get('n_classes', self.yshape[0])
        else:
            # if no n_classes specified, assumed to be regression
            # and no need for conditional inputs
            self.conditional = False

        # get other kwargs as hyperparameters
        hyperparams = {key: kwargs[key] for key in kwargs.keys() -
                       {'n_classes', 'noise_shape'}}

        # instantiating the model as the deepchem gan
        if self.conditional:
            model = dcgan.CondDCGAN(self.xshape, self.yshape, noise_shape,
                                    n_classes, hyperparams)
        else:
            model = dcgan.DCGAN(self.xshape, self.yshape, noise_shape,
                                n_classes, hyperparams)
        return model

    def generate_data(self,
                      Xs: Array,
                      Ys: Array,
                      batch_size: int):
        """
        Generating function.

        Create a batch of bootstrapped data. _train helper function.
        From deepchem tutorial 14.

        Arguments:
            Xs/Ys - training examples/targets
                type == ndarray

            batch_size - number of data points in a batch
                type == int

        Returns:
            classes - array of targets sampled from Ys
                type == ndarray

            points - array of data points sampled from Xs
                type == ndarray
        """
        # sample with replacement X, Y pairs of size batch_size
        n = len(Xs)
        indices = np.random.randomint(0, high=n, size=(batch_size,))
        classes = Xs[indices]
        points = Ys[indices]
        return classes, points

    def iterbatches(self,
                    Xs: Array,
                    Ys: Array,
                    **kwargs):
        """
        Function that creates batches of generated data.

        The deepchem fit_gan unction reads in a dictionary for training.
        This creates that dictionary for each batch. _train helper function.
        From deepchem tutorial 14.

        Arguments:
            Xs/Ys - training examples/targets
                type == ndarray

            **kwargs - Specify training hyperparameters
                    batches - number of batches to train on
                        type == int
                    batch_size - number of data points in a batch
                        type == int

        Yields:
            batched_data - data split into batches
                type == dict
        """
        # get training hyperparamters from kwargs
        batches = kwargs.get('batches', 50)
        batch_size = kwargs.get('batch_size', 32)

        # training loop
        for i in range(batches):
            classes, points = self.generate_data(Xs, Ys, batch_size)
            if len(Ys.shape) == 2:
                # Ys already one hot encoded
                pass
            else:
                # must one hot encode Ys
                classes = deepchem.metrics.to_one_hot(classes,
                                                      self.model.n_classes)
            batched_data = {self.data_inputs[0]: points,
                            self.conditional_inputs[0]: classes}
            yield batched_data

    # overridden method from UncertaintyModel class
    def _train(self,
               Xs: Array,
               Ys: Array,
               *args,  # use args thoughtfully?
               metric: Callable = None,
               **kwargs) -> Any:
        """
        Train GAN model on data.

        Arguments:
            Xs/Ys - training examples/targets
                type == ndarray

            **kwargs - keyword arguments to assign non-default training parame-
                ters or pass to nested functions.

        Returns:
            losses - array of loss for each epoch
                type == ndarray
        """
        # train GAN on data
        # self.model = deepchem GAN instance
        self._model.fit_gan(self.iterbatches(Xs, Ys, **kwargs))
        # The deepchem gan is a Keras model whose
        # outputs are [gen_loss, disrcim_loss].
        # Thus the final losses for the generator
        # and discriminator are self.model.outputs
        # This is a list of 2 KerasTensors so must evaluate it.
        losses = self.model.outputs
        # compute metric
        return losses

    # overridden method from UncertaintyModel class
    def _predict(self,
                 Xs: Array,
                 *args,
                 **kwargs):
        """
        Predict on Xs.

        Arguments:
            Xs - example data to make predictions on
                type == ndarray

            **kwargs - keyword arguments for predicting

        Returns:
            predictions - array of predictions of targets with the same length
                as Xs
                type == ndarray

            uncertainties - array of prediction uncertainties of targets with
                the same length as Xs
                type == ndarray
        """
        # adapted from deepchem tutorial 14:
        num_predictions = kwargs.get('num_predictions', 100)
        predictions = []
        if self.conditional:
            Ys = kwargs.get('Ys', None)
            assert Ys is not None, "This is a cGAN.\
                Must specify Ys (Ys=) to call predict."
            if len(Ys.shape) == 2:
                # assumes data is in bacthed form
                # Ys already one hot encoded
                one_hot_Ys = Ys
            else:
                # must one hot encode Ys
                one_hot_Ys = deepchem.metrics.to_one_hot(Ys,
                                                         self.model.n_classes)
                for i in range(num_predictions):
                    # generate data with conditional inputs
                    generated_points = self._model.predict_gan_generator(
                        conditional_inputs=[one_hot_Ys])
                    predictions.append(generated_points)
        else:
            for i in range(num_predictions):
                generated_points = self._model.predict_gan_generator()
                predictions.append(generated_points)
        # the above code generates points, but we need uncertainties as well
        predictions = np.average(predictions, axis=1)
        uncertainties = np.std(predictions, axis=1)
        return predictions, uncertainties

    def save(self, filename: str, **kwargs):
        """
        Method defined by child to save the predictor.

        Method must save into memory the object at self._model
        For other functionalities to add, see
        https://www.tensorflow.org/guide/keras/save_and_serialize

        Arguments:
            filename (str):
                name of file to save model to
        """
        # save model aka generator and discriminator separately
        if filename.endswith('.h5'):
            self._model.save(filename)
        else:
            path_to_model = filename
            self._model.save(path_to_model)
        return None

    @classmethod
    def load(cls, filename: str, **kwargs):
        """
        Method defined by child to load a predictor into memory.

        Loads the object to be assigned to self._model.
        For other functionalities to add, see
        https://www.tensorflow.org/guide/keras/save_and_serialize

        Arguments:
            filename (str):
                path of file to load
        """
        # call Keras.load function
        if filename.endswith('.h5'):
            model = tf.keras.model.load_model(filename, compile=False)
        else:
            path_to_model = filename
            model = tf.keras.model.load_model(path_to_model)
        return model
