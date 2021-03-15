"""
This class implements Deepchem's GAN class.

Deepchem's tutorial on GANs (14_Conditional_Generative_Adversarial_Networks)
can be found here:
https://github.com/deepchem/deepchem/blob/master/examples/tutorials/
    14_Conditional_Generative_Adversarial_Networks.ipynb

"""

# time
import time

# warnings
import warnings

# deep learning imports
import deepchem
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Input
from tensorflow.keras.layers import Dropout, LeakyReLU

# typing imports
from typing import Tuple, Type

# more typing
import numpy as np
Array = Type[np.ndarray]


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

    def __init__(self, xshape, yshape, noise_shape, **kwargs):
        """Override deepchem init function."""
        # These should be set by the gandy model when _build is called.
        self.xshape = xshape
        self.yshape = yshape
        self.noise_shape = noise_shape

        # base hyperparameters for generator and discirminator

        Base_hyperparams = dict(layer_dimensions=[128],
                                dropout=0.05,
                                activation='relu',
                                use_bias=True,
                                kernel_initializer="glorot_uniform",
                                bias_initializer="zeros",
                                kernel_regularizer='l2',
                                bias_regularizer=None,
                                activity_regularizer=None,
                                kernel_constraint=None,
                                bias_constraint=None)

        # Create separate hyperparam dictionaries for the generator
        # and discriminator
        self.generator_hyperparameters = Base_hyperparams.copy()
        self.discriminator_hyperparameters = Base_hyperparams.copy()

        # get network hyperparameters from kwargs
        for key in kwargs.keys():
            if key.startswith('generator_'):
                # generator param
                param = key.replace('generator_', '')
                # check if the key is a valid hyperparamter
                if param in self.generator_hyperparameters.keys():
                    self.generator_hyperparameters[param] = kwargs[key]
                else:
                    warnings.warn(f"Incorrect key {key}. Must be in\
                        {Base_hyperparams.keys()}")
            elif key.startswith('discriminator_'):
                # discriminator param
                param = key.replace('discriminator_', '')
                if param in self.discriminator_hyperparameters.keys():
                    self.discriminator_hyperparameters[param] = kwargs[key]
                else:
                    warnings.warn(f"Incorrect key {key}. Must be in\
                        {Base_hyperparams.keys()}")
            else:
                warnings.warn(f"{key} must start with generator_ or discriminator_" +
                	f"\nPassing {key} as deepchem model init kwargs.")

        # Deepchem init function + class atributes.
        super(DCGAN, self).__init__(**kwargs)

    def create_generator(self):
        """
        Create the generator as a keras model.

        kwargs contains the possible arguments for the generator.
        See Arguments.

        Other kwargs for a Dense layer can be found at
        https://keras.io/api/layers/core_layers/dense/

        Arguments:
            Kwargs for the model architecture:

            layer_dimensions - list of hidden dimension layers
                Note: This should note include the output dimension.
                Default - [128]
                type == list of ndarray
            dropout - layer dropout percetnage,
                i.e., percent of weights that are randomly set to 0
                Can choose a flooat in [0.0, 1.0)
                Default - 0.05 (5% dropout rate)
                type == float

            The kwargs for each layer that are different than the Keras
            default are:

            activation - hidden layer activation function.
                Can choose from 'relu', 'tanh', 'sigmoid', 'softmax',
                'softplus', 'softsign', 'selu', 'elu', 'exponential',
                or 'linear'. See https://keras.io/api/layers/activations/
                Default - 'relu'
                type == str
            kernel_regularizer - regularizer of kernel/ weights
                Can choose from 'l2', 'l1', etc.
                Default - 'l2'
                type == str

        Returns:
            generator - creates data from random noise
                type == Keras model

        """
        # adapted from deepchem tutorial 14:

        kwargs = self.generator_hyperparameters

        # get hyperparameters from kwargs
        layer_dimensions = kwargs.get('layer_dimensions', [128])
        dropout = kwargs.get('dropout', 0.05)
        # every other kwarg is for the layers
        layer_kwargs = {key: kwargs[key] for key in kwargs.keys()
                        - {'layer_dimensions', 'dropout'}}

        # construct input
        noise_in = Input(shape=self.get_noise_input_shape())
        # build first layer of network
        gen = Dense(layer_dimensions[0], **layer_kwargs)(noise_in)
        # adding dropout to the weights
        gen = Dropout(dropout)(gen)
        # build subsequent layers
        for layer_dim in layer_dimensions[1:]:
            gen = Dense(layer_dim, **layer_kwargs)(gen)
            gen = Dropout(dropout)(gen)

        # generator outputs
        gen = Dense(self.yshape[0], **layer_kwargs)(gen)

        # final construction of Keras model
        generator = tf.keras.Model(inputs=[noise_in],
                                   outputs=[gen])
        return generator

    def create_discriminator(self):
        """
        Create the discriminator as a keras model.

        kwargs contains the possible arguments for the discriminator.
        See Arguments.

        Other kwargs for a Dense layer can be found at
        https://keras.io/api/layers/core_layers/dense/

        Arguments:
            Kwargs for the model architecture:

            layer_dimensions - list of hidden dimension layers
                Note: This should note include the output dimension.
                Default - [128]
                type == list of ndarray
            dropout - layer dropout percetnage,
                i.e., percent of weights that are randomly set to 0
                Can choose a flooat in [0.0, 1.0)
                Default - 0.05 (5% dropout rate)
                type == float

            The kwargs for each layer that are different than the Keras
            default are:

            activation - hidden layer activation function.
                Can choose from 'relu', 'tanh', 'sigmoid', 'softmax',
                'softplus', 'softsign', 'selu', 'elu', 'exponential',
                or 'linear'. See https://keras.io/api/layers/activations/
                Default - 'relu'
                type == str
            kernel_regularizer - regularizer of kernel/ weights
                Can choose from 'l2', 'l1', etc.
                Default - 'l2'
                type == str

        Returns:
            discriminator - the discriminator outputs a probability that
            the data is real or fake
                type == Keras model

        """
        # adapted from deepchem tutorial 14:

        kwargs = self.discriminator_hyperparameters

        # get hyperparameters from kwargs
        layer_dimensions = kwargs.get('layer_dimensions', [128])
        dropout = kwargs.get('dropout', 0.05)
        # every other kwarg is for the layers
        layer_kwargs = {key: kwargs[key] for key in kwargs.keys()
                        - {'layer_dimensions', 'dropout'}}

        # construct input
        data_in = Input(shape=self.yshape)
        # build first layer of network
        discrim = Dense(layer_dimensions[0], **layer_kwargs)(data_in)
        # adding dropout to the weights
        discrim = Dropout(dropout)(discrim)
        # build subsequent layers
        for layer_dim in layer_dimensions[1:]:
            discrim = Dense(layer_dim, **layer_kwargs)(discrim)
            discrim = Dropout(dropout)(discrim)

        # To maintain the interpretation of a probability,
        # the final activation function is not a kwarg
        final_layer_kwargs = layer_kwargs.copy()
        final_layer_kwargs.update(activation='sigmoid')
        discrim_prob = Dense(1, **final_layer_kwargs)(discrim)

        # final construction of Keras model
        discriminator = tf.keras.Model(inputs=[data_in],
                                       outputs=[discrim_prob])
        return discriminator

    def fit_gan(self,
                batches,
                generator_steps=1.0,
                max_checkpoints_to_keep=5,
                checkpoint_interval=1000,
                restore=False):
        """Train this model on data.

        !! Adjusted from deepchem to return losses !!

        Parameters
        ----------
        batches: iterable
            batches of data to train the discriminator on, each
            represented as a dict that maps Inputs to values.
            It should specify values for all members of
            data_inputs and conditional_inputs.
        generator_steps: float
            the number of training steps to perform for the generator
            for each batch. This can be used to adjust the ratio of
            training steps for the generator and discriminator.
            For example, 2.0 will perform two training steps for
            every batch, while 0.5 will only perform one training step
            for every two batches.
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.
            Older checkpoints are discarded.
        checkpoint_interval: int
            the frequency at which to write checkpoints, measured in
            batches. Set this to 0 to disable automatic checkpointing.
        restore: bool
            if True, restore the model from the most recent checkpoint before
            training it.
        """
        self._ensure_built()
        gen_train_fraction = 0.0
        discrim_error = 0.0
        gen_error = 0.0
        discrim_average_steps = 0
        gen_average_steps = 0
        time1 = time.time()

        # Added HERE
        model_losses = [[], []]

        if checkpoint_interval > 0:
            manager = tf.train.CheckpointManager(self._checkpoint,
                                                 self.model_dir,
                                                 max_checkpoints_to_keep)
        for feed_dict in batches:
            # Every call to fit_generator() will increment global_step,
            # but we only want it to get incremented once for the entire
            # batch, so record the value and keep resetting it.

            global_step = self.get_global_step()

            # Train the discriminator.

            inputs = [self.get_noise_batch(self.batch_size)]
            for input in self.data_input_layers:
                inputs.append(feed_dict[input.ref()])
            for input in self.conditional_input_layers:
                inputs.append(feed_dict[input.ref()])
            discrim_error += self.fit_generator(
                [(inputs, [], [])],
                variables=self.discrim_variables,
                loss=self.discrim_loss_fn,
                checkpoint_interval=0,
                restore=restore)
            restore = False
            discrim_average_steps += 1

            # Train the generator.

            if generator_steps > 0.0:
                gen_train_fraction += generator_steps
                while gen_train_fraction >= 1.0:
                    inputs = [self.get_noise_batch(
                              self.batch_size)] + inputs[1:]
                    gen_error += self.fit_generator(
                        [(inputs, [], [])],
                        variables=self.gen_variables,
                        checkpoint_interval=0)
                    gen_average_steps += 1
                    gen_train_fraction -= 1.0
            self._global_step.assign(global_step + 1)

            # Write checkpoints and report progress.

            if discrim_average_steps == checkpoint_interval:
                manager.save()
                discrim_loss = discrim_error / max(1, discrim_average_steps)
                gen_loss = gen_error / max(1, gen_average_steps)
                print(
                    f'Step {global_step+1}: \t' +
                    f'Avg gen loss {gen_loss}, \t' +
                    f'Avg discrim loss {discrim_loss}')
                discrim_error = 0.0
                gen_error = 0.0
                discrim_average_steps = 0
                gen_average_steps = 0

                # Added HERE
                model_losses[0].append(gen_loss)
                model_losses[1].append(discrim_loss)

        # Write out final results.

        if checkpoint_interval > 0:
            if discrim_average_steps > 0 and gen_average_steps > 0:
                discrim_loss = discrim_error / discrim_average_steps
                gen_loss = gen_error / gen_average_steps
                print(
                    f'Step {global_step+1}: \t' +
                    f'Avg gen loss {gen_loss}, \t' +
                    f'Avg discrim loss {discrim_loss}')
            manager.save()
            time2 = time.time()
            print("TIMING: model fitting took %0.3f s" % (time2 - time1))

            model_losses[0].append(gen_loss)
            model_losses[1].append(discrim_loss)

        # ADDED Here
        return model_losses

    def get_noise_input_shape(self) -> Tuple[int]:
        """
        Return the shape of the noise vector.

        This should be set by the gandy model when an build is called.
        """
        return self.noise_shape

    def get_data_input_shapes(self) -> Tuple[int]:
        """
        Return the shape of the data.

        This should be set by the gandy model when an build is called.

        Data input shape is y!
        """
        return [self.yshape]

    def get_conditional_input_shapes(self) -> Array:
        """
        Return the shape of the conditional input.

        This should be set by the gandy model when an build is called.

        This is x data!
        """
        return [(self.xshape[0],)]


class CondDCGAN(DCGAN):
    """
    Conditional GAN subcless of deepchem's GAN class.

    This class is a subclass of the gans class and instead implements
    a cgan. A Conditional GAN (cGAN) has additional inputs to the
    generator and discriminator, and learns a distribution that is
    conditional on the values of those inputs. They are referred
    to as "conditional inputs".
    """

    def create_generator(self):
        """
        Create the generator as a keras model.

        kwargs contains the possible arguments for the generator.
        See Arguments.

        Other kwargs for a Dense layer can be found at
        https://keras.io/api/layers/core_layers/dense/

        Arguments:
            Kwargs for the model architecture:

            layer_dimensions - list of hidden dimension layers
                Note: This should note include the output dimension.
                Default - [128]
                type == list of ndarray
            dropout - layer dropout percetnage,
                i.e., percent of weights that are randomly set to 0
                Can choose a flooat in [0.0, 1.0)
                Default - 0.05 (5% dropout rate)
                type == float

            The kwargs for each layer that are different than the Keras
            default are:

            activation - hidden layer activation function.
                Can choose from 'relu', 'tanh', 'sigmoid', 'softmax',
                'softplus', 'softsign', 'selu', 'elu', 'exponential',
                or 'linear'. See https://keras.io/api/layers/activations/
                Default - 'relu'
                type == str
            kernel_regularizer - regularizer of kernel/ weights
                Can choose from 'l2', 'l1', etc.
                Default - 'l2'
                type == str

        Returns:
            generator - creates data from random noise
                type == Keras model

        """
        # adapted from deepchem tutorial 14:

        kwargs = self.generator_hyperparameters

        # get hyperparameters from kwargs
        layer_dimensions = kwargs.get('layer_dimensions', [128])
        dropout = kwargs.get('dropout', 0.05)
        # every other kwarg is for the layers
        layer_kwargs = {key: kwargs[key] for key in kwargs.keys()
                        - {'layer_dimensions', 'dropout'}}

        # construct input
        noise_in = Input(shape=self.get_noise_input_shape())
        conditional_in = Input(shape=self.xshape)
        gen_input = Concatenate()([noise_in, conditional_in])

        # build first layer of network
        gen = Dense(layer_dimensions[0], **layer_kwargs)(gen_input)
        # adding dropout to the weights
        gen = Dropout(dropout)(gen)
        # build subsequent layers
        for layer_dim in layer_dimensions[1:]:
            gen = Dense(layer_dim, **layer_kwargs)(gen)
            gen = Dropout(dropout)(gen)

        # generator outputs
        gen = Dense(self.yshape[0], **layer_kwargs)(gen)

        # final construction of Keras model
        generator = tf.keras.Model(inputs=[noise_in, conditional_in],
                                   outputs=[gen])
        return generator

    def create_discriminator(self):
        """
        Create the discriminator as a keras model.

        kwargs contains the possible arguments for the discriminator.
        See Arguments.

        Other kwargs for a Dense layer can be found at
        https://keras.io/api/layers/core_layers/dense/

        Arguments:
            Kwargs for the model architecture:

            layer_dimensions - list of hidden dimension layers
                Note: This should note include the output dimension.
                Default - [128]
                type == list of ndarray
            dropout - layer dropout percetnage,
                i.e., percent of weights that are randomly set to 0
                Can choose a flooat in [0.0, 1.0)
                Default - 0.05 (5% dropout rate)
                type == float

            The kwargs for each layer that are different than the Keras
            default are:

            activation - hidden layer activation function.
                Can choose from 'relu', 'tanh', 'sigmoid', 'softmax',
                'softplus', 'softsign', 'selu', 'elu', 'exponential',
                or 'linear'. See https://keras.io/api/layers/activations/
                Default - 'relu'
                type == str
            kernel_regularizer - regularizer of kernel/ weights
                Can choose from 'l2', 'l1', etc.
                Default - 'l2'
                type == str

        Returns:
            discriminator - the discriminator outputs a probability that
            the data is real or fake
                type == Keras model

        """
        # adapted from deepchem tutorial 14:

        kwargs = self.discriminator_hyperparameters

        # get hyperparameters from kwargs
        layer_dimensions = kwargs.get('layer_dimensions', [128])
        dropout = kwargs.get('dropout', 0.05)
        # every other kwarg is for the layers
        layer_kwargs = {key: kwargs[key] for key in kwargs.keys()
                        - {'layer_dimensions', 'dropout'}}
        # removing activation to implemetn LeakyReLU
        # layer_kwargs.update(activation=None)

        # construct input
        data_in = Input(shape=self.yshape)
        conditional_in = Input(shape=self.xshape,)
        discrim_input = Concatenate()([data_in, conditional_in])

        # build first layer of network
        discrim = Dense(layer_dimensions[0], **layer_kwargs)(discrim_input)
        # discrim = LeakyReLU()(discrim)
        # adding dropout to the weights
        discrim = Dropout(dropout)(discrim)
        # build subsequent layers
        for layer_dim in layer_dimensions[1:]:
            discrim = Dense(layer_dim, **layer_kwargs)(discrim)
            # discrim = LeakyReLU()(discrim)
            discrim = Dropout(dropout)(discrim)

        # To maintain the interpretation of a probability,
        # the final activation function is not a kwarg
        final_layer_kwargs = layer_kwargs.copy()
        final_layer_kwargs.update(activation='sigmoid')
        discrim_prob = Dense(1, **final_layer_kwargs)(discrim)

        # final construction of Keras model
        discriminator = tf.keras.Model(inputs=[data_in, conditional_in],
                                       outputs=[discrim_prob])
        return discriminator
