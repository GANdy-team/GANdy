## imports
import gandy.models.models
import tensorflow as tf

class bnn(gandy.models.models.UncertaintyModel):
    '''
    Implements a Bayesian Neural Network (BNN)
    BNNS place a prior on the weights of the network and apply Bayes rule
    '''

    def create_model_inputs(self, feature_names):
        '''
        Arguments:
            feature_names - example data to make predictions on
                type == ndarray, list, or dictionary      
        Returns:
            inputs 
                type == dictionary
        '''
        # do something like:
        # (from https://keras.io/examples/keras_recipes/bayesian_neural_networks/)
        # inputs = {}
        # for feature_name in feature_names:
        #     inputs[feature_name] = tf.keras.layers.Input(
        #         name=feature_name, shape=(1,), dtype=tf.float32
        #     )
        # return inputs

    # overridden method from UncertaintyModel class
    def _build(self, **kwargs): 
        '''
        Construct the model
        '''
        # do something like:
        # (from https://keras.io/examples/keras_recipes/bayesian_neural_networks/)
        # feature_names = **kwargs
        # or default feature_names = np.arange(xshape[0])
        # activation, optimizer, loss = **kwargs
        # or default activation = 'relu', optimizer = tf.keras.optimizers.adam, loss = tf.keras.losses.MSE
        # inputs = create_model_inputs(feature_names)
        # input_values = [value for _, value in sorted(inputs.items())]
        # features = tf.keras.layers.concatenate(input_values)
        # features = tf.keras.layers.BatchNormalization()(features)

        # Create hidden layers with deterministic weights using the Dense layer.
        # for units in hidden_units:
        #     features = tf.keras.layers.Dense(units, activation=activation)(features)
        # The output is deterministic: a single point estimate.
        # outputs = layers.Dense(units=1)(features)

        # model = keras.Model(inputs=inputs, outputs=outputs)
        # model.compile(optimizer=optimizer, loss=loss)
        # self.model = model
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

        return predictions, uncertainties