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
# import tensorflow as tf

# typing imports
from typing import Any, Object, Type

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
        # setting the dcgan global variables
        dcgan.XSHAPE = self.xshape
        dcgan.YSHAPE = self.yshape
        # get noise shape from kwargs
        # default is 10 dimensional
        self.noise_shape = kwargs.get('noise_shape', (10,))
        dcgan.NOISE_SHAPE = self.noise_shape

        # determine whether to use gan or conditional gan
        if len(self.yshape) == 2:
            self.conditional = True
            # get n_classes from kwargs
            # default is the y dimension
            # e.g., regression would be == 1
            # This would also be correct for a one hot encoded y vector.
            self.n_classes = kwargs.get('n_classes', self.yshape[0])
            dcgan.N_CLASSES = self.n_classes
        else:
            self.conditional = False

        # instantiating the model as the deepchem gan
        if self.conditional:
            model = dcgan.CondDCGAN(**kwargs)
        else:
            model = dcgan.DCGAN(**kwargs)
        return model

    def generate_data(self,
                      Xs: Array,
                      Ys: Array,
                      batch_size: int):
        """
        Generating function.

        Create a batch of bootstrapped data. _train helper function

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

    def iterate_batches(self,
                        Xs: Array,
                        Ys: Array,
                        **kwargs):
        """
        Function that creates batches of generated data.

        The deepchem fit_gan unction reads in a dictionary for training.
        This creates that dictionary for each batch. _train helper function

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
            classes = deepchem.metrics.to_one_hot(classes,
                                                  self.model.N_CLASSES)
            batched_data = {self.data_inputs[0]: points,
                            self.conditional_inputs[0]: classes}
            yield batched_data

    # overridden method from UncertaintyModel class
    def _train(self,
               Xs: Array,
               Ys: Array,
               *args,
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
        self.model.fit_gan(self.iterbatches(Xs, Ys, **kwargs))
        # The deepchem gan is a Keras model whose
        # outputs are [gen_loss, disrcim_loss].
        # Thus the final losses for the generator
        # and discriminator are self.model.outputs
        # This is a list of 2 KerasTensors so must evaluate it.
        losses = self.model.outputs
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
        if self.conditional:
            Ys = kwargs.get('Ys', None)
            assert Ys is not None, "This is a cGAN. Must specify Ys (Ys=) to call predict."
            one_hot_Ys = deepchem.metrics.to_one_hot(Ys, self.model.N_CLASSES)
            generated_points = self.predict_gan_generator(
                conditional_inputs=[one_hot_Ys])
        else:
            generated_points = self.predict_gan_generator()
        # the above code generates points, but we need uncertainties as well
        predictions, uncertainties = generated_points, None
        return predictions, uncertainties

    def _save(self, filename: str, **kwargs):
        """
        Method defined by child to save the predictor.

        Method must save into memory the object at self._model

        Arguments:
            filename (str):
                name of file to save model to
        """
        # save model aka generator and discriminator separately
        # assert filename.endswith('.h5') or other extension
        # self.generator.save(filename)
        return None

    def _load(self, filename: str, **kwargs):
        """
        Method defined by child to load a predictor into memory.

        Loads the object to be assigned to self._model.

        Should this be a class method?

        Arguments:
            filename (str):
                path of file to load
        """
        # call Keras.load function
        # two filenames, one for gen and one for discrim?
        # model = tf.keras.model.load_model(filename, compile=False)
        model = None
        return model
