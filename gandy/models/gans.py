## imports
import gandy.models.models
import deepchem
import tensorflow as tf

'''
Deepchem's tutorial on GANs (14_Conditional_Generative_Adversarial_Networks) can be found here:
https://github.com/deepchem/deepchem/blob/master/examples/tutorials/14_Conditional_Generative_Adversarial_Networks.ipynb
'''

class gans(deepchem.models.GAN, gandy.models.models.UncertaintyModel):
    '''
    Implements Generative Adversarial Networks.
    A Generative Adversarial Network (GAN) is a type of generative model.  It
    consists of two parts called the "generator" and the "discriminator".  The
    generator takes random noise as input and transforms it into an output that
    (hopefully) resembles the training data.  The discriminator takes a set of
    samples as input and tries to distinguish the real training samples from the
    ones created by the generator.  Both of them are trained together.  The
    discriminator tries to get better and better at telling real from false data,
    while the generator tries to get better and better at fooling the discriminator.
    Thank you to deepchem at
    https://github.com/deepchem/deepchem/blob/master/deepchem/models/gan.py#L14-L442
    for the information about GANS. This class builds off of the deepchem GAN class.
    '''

    def __init__(self, xshape, yshape, **kwargs):
        '''
        Initializes instance of a GAN
        '''
        # the MRO order of init calls is deepchem.models.GAN first, then gandy.models.models.UncertaintyModel
        super(gans, self).__init__(xshape=xshape, yshape=yshape, **kwargs)

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
        # gen_dense_lay_1 = Dense(layer_one_dimension, activation=tf.nn.relu)(noise_in)
        # gen_outputs = Dense(output_layer_dimension, activation=tf.nn.relu)(gen_dense_lay_1)
        # we should make above code a for loop so that num_layers is a changeable parameter
        # self.generator = tf.keras.Model(inputs=[noise_in], outputs=[gen_outputs])
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
        # discrim_lay_1 = Dense(layer_one_dimension, activation=tf.nn.relu)(data_in)
        # discrim_prob = Dense(1, activation=tf.sigmoid)(discrim_lay_1)
        # self.discriminator = tf.keras.Model(inputs=[data_in], outputs=[discrim_prob])
        return None

    def get_noise_input_shape(self, **kwargs):
        '''
        Returns the shape of the noise vector
        '''
        return noise.shape

    def get_data_input_shapes(self **kwargs):
    	'''
        Returns the shape of the data, which should be xshape
        '''
        return self.xshape

    # overridden method from UncertaintyModel class
    def _build(self, **kwargs): 
        '''
        Construct the model
        '''
        # do something like:
        # self.create_generator(**kwargs)
        # self.create_discriminator(**kwargs)
        # self.n_classes = self.yshape
        return None

    # overridden method from UncertaintyModel class
    def _train(self, Xs, Ys, **kwargs):
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
    def _predict(self, Xs, **kwargs):
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
        # generated_points = gan.predict_gan_generator(conditional_inputs=[one_hot_Ys])
        # the above code generates points, but we need uncertainties as well
        return predictions, uncertainties



class cgans(gans):
    '''
    This class is a subclass of the gans class and instead implements a cgan.
    A Conditional GAN (cGAN) has additional inputs to the generator and discriminator,
    and learns a distribution that is conditional on the values of those inputs. They
    are referred to as "conditional inputs".
    '''
    
    def __init__(self, xshape, yshape, **kwargs):
        '''
        Initializes instance of a cGAN
        '''
        # set special conditional properties here
        super(cgans, self).__init__(xshape, yshape,, **kwargs)
        return # nothing? do we need return statements in the init?

    def get_conditional_input_shapes(self, **kwargs):
        '''
        Returns the shape of the conditional input in which the CGAN learns a distribution
        '''
        # adapted from deepchem tutorial 14:
        return [(self.n_classes,)]

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
        # conditional_in = Input(shape=(self.n_classes,))
        # gen_input = Concatenate()([noise_in, conditional_in])
        # gen_dense_lay_1 = Dense(layer_one_dimension, activation=tf.nn.relu)(gen_input)
        # gen_outputs = Dense(output_layer_dimension, activation=tf.nn.relu)(gen_dense_lay_1)
        # self.generator = tf.keras.Model(inputs=[noise_in, conditional_in], outputs=[gen_outputs])
        return self.generator

    def create_discriminator(self, **kwargs):
        '''
        Creates the discriminator (as a keras model)
        Saves self.discriminator as this model
        '''
        # adapted from deepchem tutorial 14:
        # do something like:
        # data_in = Input(shape=(output_layer_dimension,))
        # conditional_in = Input(shape=(self.n_classes,))
        # discrim_in = Concatenate()([data_in, conditional_in])
        # discrim_lay_1 = Dense(layer_one_dimension, activation=tf.nn.relu)(discrim_in)
        # discrim_prob = Dense(1, activation=tf.sigmoid)(discrim_lay_1)
        # self.discriminator = tf.keras.Model(inputs=[data_in, conditional_in], outputs=[discrim_prob])
        return self.discriminator