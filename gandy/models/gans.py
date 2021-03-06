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
    This class builds off of the deepchem GAN class.
    """

    # overridden method from UncertaintyModel class
    def _build(self, **kwargs) -> Object:
        """
        Construct the model.

        This instantiates the deepchem gan as the model.
        """
        # setting the dcgan global variables
        dcgan.XSHAPE = self.xshape
        dcgan.YSHAPE = self.yshape
        # get noise shape from kwargs
        # default noise is (10,)
        dcgan.NOISE_SHAPE = kwargs.get('noise_shape', (10,))
        # determine whether to use gan or condition gan
        if len(yshape) == 3:
            conditional = True
        else:
            conditional = False
        # instantiating the model as the deepchem gan
        if conditional:
            model = dcgan.CondDCGAN(**kwargs)
        else:
            model = dcgan.DCGAN(**kwargs)
        return model

    def generate_data(Xs: Array,
                        Ys: Array,
                        batch_size: int):
        """
        Generating function.

        Creates a batch of bootstrapped data. _train helper function

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
        # sample with replacement a batch size num of x, y pairs
        classes, points = None
        return classes, points

    def iterate_batches(Xs: Array,
                        Ys: Array,
                        epcohs: int):
        """
        Function that creates batches of generated data.

        The deepchem fit_gan unction reads in a dictionary for training.
        This creates that dictionary for each batch. _train helper function

        Arguments:
            Xs/Ys - training examples/targets
                type == ndarray

            batch_size - number of data points in a batch
                type == int

        Yields:
            batched_data - data split into batches
                type == dict
        """
        # for i in range(batches):
        #     classes, points = generate_data(self.batch_size)
        #     classes = deepchem.metrics.to_one_hot(classes, n_classes)
        #     batched_data = {self.data_inputs[0]: points,
        #                     self.conditional_inputs[0]: classes}
        #     yield batched_data


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
        """
        # epochs and batch_size in args
        # self.batch_size = batch_size
        # self.fit_gan(iterbatches(Xs, Ys, epochs))
        # losses = self.model.outputs
        losses = None
        return losses

    # overridden method from UncertaintyModel class
    def _predict(self,
                 Xs: Array,
                 *args,
                 **kwargs):
        """
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
        # pseudocode
        # adapted from deepchem tutorial 14:
        # one_hot_Ys = deepchem.metrics.to_one_hot(Ys, self.n_classes)
        # generated_points = self.predict_gan_generator(
        #                                 conditional_inputs=[one_hot_Ys])
        # the above code generates points, but we need uncertainties as well
        predictions, uncertainties = None, None
        return predictions, uncertainties

    def _save(filename: str, **kwargs):
        """
        Method defined by child to save the predictor.

        Method must save into memory the object at self._model

        Args:
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

        Args:
            filename (str):
                path of file to load
        """
        # call Keras.load function
        # two filenames, one for gen and one for discrim?
        # model = tf.keras.model.load_model(filename, compile=False)
        model = None
        return model
