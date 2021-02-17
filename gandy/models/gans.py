## imports



class gans(UncertaintyModel):
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
        # do something like:
        # call build or create generator beforehand
        # to set the generator and discriminator as attributes
        # potentially wrap gen/dis together into one model (below)
        # model = tf.keras.Model(inputs=inputs, outputs=outputs)
        # make sure UncertaintyModel is within scope
        super(gans, self).__init__(xshape, yshape, **kwargs)
        return


    def create_generator(self, **kwargs):
        '''
        Creates the generator (as a keras model)
        Saves self.generator as this model
        '''
        # do something like:
        # self.generator = keras model
        return

    def create_discriminator(self, **kwargs):
        '''
        Creates the discriminator (as a keras model)
        Saves self.discriminator as this model
        '''
        # do something like:
        # self.discriminator = keras model
        return

    def add_loss(self, **kwargs):
        '''
        Adds an appropriate GAN loss to the model
        '''

        return

    # overridden method from parent class
    def build(self, **kwargs): 
        '''
        Construct the model
        '''
        # do something like:
        # model = generator + discriminator
        # or call super().build()
        return

    def get_noise_input_shape(self, **kwargs):
        '''
        Returns the shape of the noise vector
        '''
        # find noise from model architecture
        return noise.shape



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
        return

    def get_conditional_input_shape(self, **kwargs):
        '''
        Returns the shape of the conditional input in which the CGAN learns a distribution
        '''

        return