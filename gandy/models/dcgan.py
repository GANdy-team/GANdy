"""
This class implements Deepchem's GAN class.

Deepchem's tutorial on GANs (14_Conditional_Generative_Adversarial_Networks)
can be found here:
https://github.com/deepchem/deepchem/blob/master/examples/tutorials/
    14_Conditional_Generative_Adversarial_Networks.ipynb

"""

# deep learning imports
import deepchem
import tensorflow as tf
from tf.keras.layers import Concatenate, Dense, Dropout, Input

# typing imports
from typing import Tuple, Object, Type

# more typing
import numpy as np
Array = Type[np.ndarray]

# These should be set by the gandy model when _build is called.
XSHAPE = None
YSHAPE = None
NOISE_SHAPE = None
N_CLASSES = None


class DCGAN(deepchem.models.GAN):
    """
    Implement Generative Adversarial Networks.

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

    This class builds off of the deepchem GAN class found at the url above.
    """

    def create_generator(self, **kwargs):
        """
        Create the generator as a keras model.

        kwargs contains the possible arguments for the generator.
        See Arguments.

        Other kwargs for a Dense layer can be found at
        https://keras.io/api/layers/core_layers/dense/

        Arguments:
            layer_dimensions - list of hidden dimension layers
                Note: This should note include the output dimension.
                Default - [128]
                type == list of ndarray
            activation - hidden layer activation function.
                Can choose from 'relu', 'tanh', 'sigmoid', 'softmax',
                'softplus', 'softsign', 'selu', 'elu', 'exponential',
                or 'linear'. See https://keras.io/api/layers/activations/
                Default - 'relu'
                type == str
            kernel_regularizer - regularizer of kernel/ weights
                Can choose from 'l2', 'l1'
                Default - 'l2'
                type == str
            dropout - layer dropout percetnage,
                i.e., percent of weights that are randomly set to 0
                Can choose a flooat in [0.0, 1.0)
                Default - 0.05 (5% dropout rate)
                type == float

        Returns:
            generator - the discriminator outputs a probability that
            the data is real or fake
                type == Keras model

        """
        # adapted from deepchem tutorial 14:

        # get hyperparameters from kwargs
        layer_dimensions = kwargs.get('layer_dimensions', [128])
        activation = kwargs.get('activation', 'relu')
        kernel_regularizer = kwargs.get('kernel_regularizer', 'l2')
        dropout = kwargs.get('dropout', 0.05)

        # construct input
        noise_in = Input(shape=self.get_noise_input_shape())
        # build first layer of network
        gen = Dense(layer_dimensions[0], activation=activation,
                    kernel_regularizer=kernel_regularizer)(noise_in)
        # adding dropout to the weights
        gen = Dropout(dropout)(gen)
        # build subsequent layers
        for layer_dim in layer_dimensions[1:]:
            gen = Dense(layer_dim, activation=activation)(gen)
            gen = Dropout(dropout)(gen)

        # generator outputs
        gen = Dense(XSHAPE[0], activation=activation)(gen)
        gen = Dropout(dropout)(gen)

        # final construction of Keras model
        generator = tf.keras.Model(inputs=[noise_in],
                                   outputs=[gen])
        return generator

    def create_discriminator(self, **kwargs):
        """
        Create the discriminator as a keras model.

        kwargs contains the possible arguments for the discriminator.
        See Arguments.

        Other kwargs for a Dense layer can be found at
        https://keras.io/api/layers/core_layers/dense/

        Arguments:
            layer_dimensions - list of hidden dimension layers
                Default - [128]
                type == list of ndarray
            activation - hidden layer activation function.
                Can choose from 'relu', 'tanh', 'sigmoid', 'softmax',
                'softplus', 'softsign', 'selu', 'elu', 'exponential',
                or 'linear'. See https://keras.io/api/layers/activations/
                Default - 'relu'
                type == str
            kernel_regularizer - regularizer of kernel/ weights
                Can choose from 'l2', 'l1'
                Default - 'l2'
                type == str
            dropout - layer dropout percetnage,
                i.e., percent of weights that are randomly set to 0
                Can choose a flooat in [0.0, 1.0)
                Default - 0.05 (5% dropout rate)
                type == float

        Returns:
            discriminator - the discriminator outputs a probability that
            the data is real or fake
                type == Keras model

        """
        # adapted from deepchem tutorial 14:

        # get hyperparameters from kwargs
        layer_dimensions = kwargs.get('layer_dimensions', [128])
        activation = kwargs.get('activation', 'relu')
        kernel_regularizer = kwargs.get('kernel_regularizer', 'l2')
        dropout = kwargs.get('dropout', 0.05)

        # construct input
        data_in = Input(shape=XSHAPE)
        # build first layer of network
        discrim = Dense(layer_dimensions[0], activation=activation,
                        kernel_regularizer=kernel_regularizer)(data_in)
        # adding dropout to the weights
        discrim = Dropout(dropout)(discrim)
        # build subsequent layers
        for layer_dim in layer_dimensions[1:]:
            discrim = Dense(layer_dim, activation=activation)(discrim)
            discrim = Dropout(dropout)(discrim)

        # To maintain the interpretation of a probability,
        # the final activation function is not a kwarg
        discrim_prob = Dense(1, activation='sigmoid')(discrim)

        # final construction of Keras model
        discriminator = tf.keras.Model(inputs=[data_in],
                                       outputs=[discrim_prob])
        return discriminator

    def get_noise_input_shape(self) -> Tuple[int]:
        """
        Return the shape of the noise vector.

        This should be set by the gandy model when an build is called.
        """
        return NOISE_SHAPE

    def get_data_input_shapes(self) -> Tuple[int]:
        """
        Return the shape of the data.

        This should be set by the gandy model when an build is called.
        """
        return XSHAPE


class CondDCGAN(DCGAN):
    """
    Conditional GAN subcless of deepchem's GAN class.

    This class is a subclass of the gans class and instead implements
    a cgan. A Conditional GAN (cGAN) has additional inputs to the
    generator and discriminator, and learns a distribution that is
    conditional on the values of those inputs. They are referred
    to as "conditional inputs".
    """

    def get_conditional_input_shapes(self, **kwargs) -> Array:
        """
        Return the shape of the conditional input.

        This should be set by the gandy model when an build is called.
        """
        return [(N_CLASSES,)]

    def create_generator(self, **kwargs) -> Object:
        """
        Create the generator as a keras model.

        kwargs contains the possible arguments for the generator.
        See Arguments.

        Other kwargs for a Dense layer can be found at
        https://keras.io/api/layers/core_layers/dense/

        Arguments:
            layer_dimensions - list of hidden dimension layers
                Note: This should note include the output dimension.
                Default - [128]
                type == list of ndarray
            activation - hidden layer activation function.
                Can choose from 'relu', 'tanh', 'sigmoid', 'softmax',
                'softplus', 'softsign', 'selu', 'elu', 'exponential',
                or 'linear'. See https://keras.io/api/layers/activations/
                Default - 'relu'
                type == str
            kernel_regularizer - regularizer of kernel/ weights
                Can choose from 'l2', 'l1'
                Default - 'l2'
                type == str
            dropout - layer dropout percetnage,
                i.e., percent of weights that are randomly set to 0
                Can choose a flooat in [0.0, 1.0)
                Default - 0.05 (5% dropout rate)
                type == float

        Returns:
            generator - the discriminator outputs a probability that
            the data is real or fake
                type == Keras model

        """
        # adapted from deepchem tutorial 14:

        # get hyperparameters from kwargs
        layer_dimensions = kwargs.get('layer_dimensions', [128])
        activation = kwargs.get('activation', 'relu')
        kernel_regularizer = kwargs.get('kernel_regularizer', 'l2')
        dropout = kwargs.get('dropout', 0.05)

        # construct input
        noise_in = Input(shape=self.get_noise_input_shape())
        conditional_in = Input(shape=(N_CLASSES,))
        gen_input = Concatenate()([noise_in, conditional_in])

        # build first layer of network
        gen = Dense(layer_dimensions[0], activation=activation,
                    kernel_regularizer=kernel_regularizer)(gen_input)
        # adding dropout to the weights
        gen = Dropout(dropout)(gen)
        # build subsequent layers
        for layer_dim in layer_dimensions[1:]:
            gen = Dense(layer_dim, activation=activation)(gen)
            gen = Dropout(dropout)(gen)

        # generator outputs
        gen = Dense(XSHAPE[0], activation=activation)(gen)
        gen = Dropout(dropout)(gen)

        # final construction of Keras model
        generator = tf.keras.Model(inputs=[noise_in, conditional_in],
                                   outputs=[gen])
        return generator

    def create_discriminator(self, **kwargs) -> Object:
        """
        Create the discriminator as a keras model.

        kwargs contains the possible arguments for the discriminator.
        See Arguments.

        Other kwargs for a Dense layer can be found at
        https://keras.io/api/layers/core_layers/dense/

        Arguments:
            layer_dimensions - list of hidden dimension layers
                Default - [128]
                type == list of ndarray
            activation - hidden layer activation function.
                Can choose from 'relu', 'tanh', 'sigmoid', 'softmax',
                'softplus', 'softsign', 'selu', 'elu', 'exponential',
                or 'linear'. See https://keras.io/api/layers/activations/
                Default - 'relu'
                type == str
            kernel_regularizer - regularizer of kernel/ weights
                Can choose from 'l2', 'l1'
                Default - 'l2'
                type == str
            dropout - layer dropout percetnage,
                i.e., percent of weights that are randomly set to 0
                Can choose a flooat in [0.0, 1.0)
                Default - 0.05 (5% dropout rate)
                type == float

        Returns:
            discriminator - the discriminator outputs a probability that
            the data is real or fake
                type == Keras model

        """
        # adapted from deepchem tutorial 14:

        # get hyperparameters from kwargs
        layer_dimensions = kwargs.get('layer_dimensions', [128])
        activation = kwargs.get('activation', 'relu')
        kernel_regularizer = kwargs.get('kernel_regularizer', 'l2')
        dropout = kwargs.get('dropout', 0.05)

        # construct input
        data_in = Input(shape=XSHAPE)
        conditional_in = Input(shape=(N_CLASSES,))
        discrim_input = Concatenate()([data_in, conditional_in])

        # build first layer of network
        discrim = Dense(layer_dimensions[0], activation=activation,
                        kernel_regularizer=kernel_regularizer)(discrim_input)
        # adding dropout to the weights
        discrim = Dropout(dropout)(discrim)
        # build subsequent layers
        for layer_dim in layer_dimensions[1:]:
            discrim = Dense(layer_dim, activation=activation)(discrim)
            discrim = Dropout(dropout)(discrim)

        # To maintain the interpretation of a probability,
        # the final activation function is not a kwarg
        discrim_prob = Dense(1, activation='sigmoid')(discrim)

        # final construction of Keras model
        discriminator = tf.keras.Model(inputs=[data_in, conditional_in],
                                       outputs=[discrim_prob])
        return discriminator
