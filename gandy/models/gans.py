'''
Deepchem's tutorial on GANs (14_Conditional_Generative_Adversarial_Networks)
can be found here:
https://github.com/deepchem/deepchem/blob/master/examples/tutorials/
    14_Conditional_Generative_Adversarial_Networks.ipynb
'''

# gandy imports
import gandy.models.models

# deep learning imports
import deepchem
import tensorflow as tf

# typing imports
from typing import Tuple, Iterable, Any, Object, Type

# typing
Array = Type[numpy.ndarray]


class gan(deepchem.models.GAN, gandy.models.models.UncertaintyModel):
    '''
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
    '''

    def __init__(self,
                 xshape: Tuple[int],
                 yshape: Tuple[int],
                 **kwargs):
        '''
        Initializes instance of a GAN
        '''
        # the MRO order of init calls is deepchem.models.GAN first,
        # then gandy.models.models.UncertaintyModel
        super(gan, self).__init__(xshape=xshape, yshape=yshape, **kwargs)

    def create_generator(self, **kwargs):
        '''
        Creates the generator (as a keras model)
        Saves self.generator as this model
        '''
        # adapted from deepchem tutorial 14:
        # do something like:
        # hyperparameters = **kwargs
        # output_layer_dimension = self.xshape[0]
        # noise_in = Input(shape=get_noise_input_shape())
        # gen_dense_lay_1 = Dense(layer_one_dimension,
        #                 activation=kwargs.activation)(noise_in)
        # gen_outputs = Dense(output_layer_dimension,
        #                 activation=kwargs.activation)(gen_dense_lay_1)
        # make above code for loop s.t. num_layers is changeable parameter
        # self.generator = tf.keras.Model(inputs=[noise_in],
        #                 outputs=[gen_outputs])
        return None

    def create_discriminator(self, **kwargs):
        '''
        Creates the discriminator (as a keras model)
        Saves self.discriminator as this model
        '''
        # adapted from deepchem tutorial 14:
        # do something like:
        # hyperparameters = **kwargs
        # data_in = Input(shape=(output_layer_dimension,))
        # discrim_lay_1 = Dense(layer_one_dimension,
        #                 activation=activation)(data_in)
        # discrim_prob = Dense(1, activation=tf.sigmoid)(discrim_lay_1)
        # self.discriminator = tf.keras.Model(inputs=[data_in],
        #                 outputs=[discrim_prob])
        return None

    def get_noise_input_shape(self, **kwargs) -> Tuple[int]:
        '''
        Returns the shape of the noise vector
        '''
        return noise.shape

    def get_data_input_shapes(self, **kwargs) -> Tuple[int]:
        '''
        Returns the shape of the data, which should be xshape
        '''
        return self.xshape

    # overridden method from UncertaintyModel class
    def _build(self, **kwargs) -> Object:
        '''
        Construct the model
        '''
        # do something like:
        # self.create_generator(**kwargs)
        # self.create_discriminator(**kwargs)
        # self.n_classes = self.yshape
        return {'generator': self.generator,
                'discriminator': self.discriminator}

    # overridden method from UncertaintyModel class
    def _train(self,
               Xs: Array,
               Ys: Array,
               *args,
               **kwargs) -> Any:
        '''
        Trains GAN model on data

        Arguments:
            Xs/Ys - training examples/targets
                type == ndarray

            **kwargs - keyword arguments to assign non-default training parame-
                ters or pass to nested functions.
        '''

        return losses

    # overridden method from UncertaintyModel class
    def _predict(self,
                 Xs: Array,
                 *args,
                 **kwargs):
        '''
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
        '''
        # pseudocode
        # adapted from deepchem tutorial 14:
        # one_hot_Ys = deepchem.metrics.to_one_hot(Ys, self.n_classes)
        # generated_points = gan.predict_gan_generator(
        #                                 conditional_inputs=[one_hot_Ys])
        # the above code generates points, but we need uncertainties as well
        return predictions, uncertainties

    def _save(filename: str, **kwargs):
        """Method defined by child to save the predictor.

        Method must save into memory the object at self.model

        Args:
            filename (str):
                name of file to save model to
        """
        # save model aka generator and discriminator separately
        return None

    def _load(self, filename: str, **kwargs):
        """Method defined by child to load a predictor into memory.

        Loads the object to be assigned to self.model.

        Args:
            filename (str):
                path of file to load
        """
        # call Keras.load function
        return model


class cgan(gan):
    '''
    This class is a subclass of the gans class and instead implements
    a cgan. A Conditional GAN (cGAN) has additional inputs to the
    generator and discriminator, and learns a distribution that is
    conditional on the values of those inputs. They are referred
    to as "conditional inputs".
    '''

    def get_conditional_input_shapes(self, **kwargs) -> Array:
        '''
        Returns the shape of the conditional input
        in which the CGAN learns a distribution
        '''
        # adapted from deepchem tutorial 14:
        return [(self.n_classes,)]

    def create_generator(self, **kwargs) -> Object:
        '''
        Creates the generator (as a keras model)
        Saves self.generator as this model
        '''
        # adapted from deepchem tutorial 14:
        # do something like:
        # hyperparameters = **kwargs
        # output_layer_dimension = self.xshape[0]
        # noise_in = Input(shape=get_noise_input_shape())
        # conditional_in = Input(shape=(self.n_classes,))
        # gen_input = Concatenate()([noise_in, conditional_in])
        # gen_dense_lay_1 = Dense(layer_one_dimension,
        #                             activation=activation)(gen_input)
        # gen_outputs = Dense(output_layer_dimension,
        #                             activation=acitvation)(gen_dense_lay_1)
        # self.generator = tf.keras.Model(
        #             inputs=[noise_in, conditional_in], outputs=[gen_outputs])
        return self.generator

    def create_discriminator(self, **kwargs) -> Object:
        '''
        Creates the discriminator (as a keras model)
        Saves self.discriminator as this model
        '''
        # adapted from deepchem tutorial 14:
        # do something like:
        # data_in = Input(shape=(output_layer_dimension,))
        # conditional_in = Input(shape=(self.n_classes,))
        # discrim_in = Concatenate()([data_in, conditional_in])
        # discrim_lay_1 = Dense(layer_one_dimension,
        #                 activation=activation)(discrim_in)
        # discrim_prob = Dense(1, activation=tf.sigmoid)(discrim_lay_1)
        # self.discriminator = tf.keras.Model(
        #            inputs=[data_in, conditional_in], outputs=[discrim_prob])
        return self.discriminator
