"""
Bayes NN.

This contains the Bayes NN class, based on the KEras tutorial at
https://keras.io/examples/keras_recipes/bayesian_neural_networks/
"""

# imports
import gandy.models.models
# import tensorflow as tf

# typing imports
from typing import Any, Callable, Type

# typing
import numpy
Array = Type[numpy.ndarray]


class BNN(gandy.models.models.UncertaintyModel):
    """
    Implements a Bayesian Neural Network (BNN)
    BNNS place a prior on the weights of the network and apply Bayes rule.
    The object of the Bayesian approach for modeling neural networks is to
    capture the epistemic uncertainty, which is uncertainty about the model
    fitness, due to limited training data.

    The idea is that, instead of learning specific weight (and bias) values
    in the neural network, the Bayesian approach learns weight distributions
    - from which we can sample to produce an output for a given input - to
    encode weight uncertainty.

    Thank you to
    https://keras.io/examples/keras_recipes/bayesian_neural_networks/
    for a guide to implementing a BNN with Keras.
    """

    def prior(kernel_size, bias_size, dtype=None) -> Callable:
        '''
        Arguments:
            kernel_size
                type == float or int
            bias_size
                type == float or int
        Returns:
            prior model
                type == Keras sequential model
        '''
        # from keras tutorial:
        # Note: this is hard-coded to be unit normal!
        # n = kernel_size + bias_size
        # prior_model = keras.Sequential(
        #     [
        #         tfp.layers.DistributionLambda(
        #             lambda t: tfp.distributions.MultivariateNormalDiag(
        #                 loc=tf.zeros(n), scale_diag=tf.ones(n)
        #             )
        #         )
        #     ]
        # )
        prior_model = None
        return prior_model

    # Define variational posterior weight distribution as multivariate
    # Gaussian. Note that the learnable parameters for this
    # distribution are the means, variances, and covariances.
    def posterior(kernel_size, bias_size, dtype=None) -> Callable:
        '''
        Arguments:
            kernel_size
                type == float or int
            bias_size
                type == float or int
        Returns:
            posterior model
                type == Keras sequential model
        '''
        # n = kernel_size + bias_size
        # posterior_model = keras.Sequential(
        #     [
        #         tfp.layers.VariableLayer(
        #            tfp.layers.MultivariateNormalTriL.params_size(n),
        #            dtype=dtype
        #         ),
        #         tfp.layers.MultivariateNormalTriL(n),
        #     ]
        # )
        posterior_model = None
        return posterior_model

    # Since the output of the model is a distribution, rather than a
    # point estimate, we use the negative loglikelihood as our loss function
    # to compute how likely to see the true data (targets) from the
    # estimated distribution produced by the model.
    def negative_loglikelihood(targets, estimated_distribution) -> Array:
        '''
        Arguments:
            targets - training targets
                type == ndarray
            estimated_distribution -
                type == function that has a log probability (keras loss e.g.)
        Returns:
            negative log likelihood
                type == ndarray
        '''
        # do something like:
        # https://keras.io/examples/keras_recipes/bayesian_neural_networks/
        # return -estimated_distribution.log_prob(targets)

    # overridden method from UncertaintyModel class
    def _build(self, *args, **kwargs) -> Callable:
        '''
        Construct the model.
        User has the option to specify:
            optional params in args:
            (features=None, units=[10], activation='relu' = args*)
            - feature names (default = column number)
            - hidden unit layer size (default = [10])
            - activation (default = 'relu')
            kwargs can include:
            - loss (default = relu)
            - optimizer (default = adam)
            or anything needed to compile model
            (think about default vals for required params)
        '''
        # do something like:
        # https://keras.io/examples/keras_recipes/bayesian_neural_networks/

        # if features is None:
        #     features = np.arange(xshape[0])

        # default activation = 'relu'
        # default optimizer = tf.keras.optimizers.adam
        # default loss = tf.keras.losses.MSE

        # # making appropriate loss:
        # estimated_distribution = loss
        # make this a hyperparamter or Gaussian?
        # loss = negative_loglikelihood(targets, estimated_distribution)
        # get train_size, i.e., train_size = xshape[0]

        # inputs = keras.Input(self.xshape)
        # input_values = list(inputs.values())
        # features = tf.keras.layers.concatenate(input_values)
        # features = tf.keras.layers.BatchNormalization()(features)

        # Deterministic BNNs = layer weights using Dense layers whereas
        # Probabilistic BNNs = layer weights using DenseVariational layers.
        # for unit in units:
        #   features = tfp.layers.DenseVariational(
        #         units=unit,
        #         make_prior_fn=self.prior,
        #         make_posterior_fn=self.posterior,
        #         kl_weight=1 / train_size,
        #         activation=activation,
        #     )(features)

        # Create a probabilistic output (Normal distribution),
        # and use the Dense layer to produce the parameters of
        # the distribution.
        # We set units=2 to learn both the mean and the variance of the
        # Normal distribution.
        # distribution_params = layers.Dense(units=2)(features)
        # outputs = tfp.layers.IndependentNormal(1)(distribution_params)

        # model = keras.Model(inputs=inputs, outputs=outputs)
        # model.compile(**kwargs)
        model = None
        return model

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
        # losses = self.model.fit(Xs, **kwargs)
        losses = None
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
        # mean, std = self.model.evaluate(Xs, **kwargs)
        # BNN model returns mean and variance as output
        # convert to predictions and uncertainties
        predictions, uncertainties = None, None
        return predictions, uncertainties

    def save(filename: str, **kwargs):
        """Method defined by child to save the predictor.

        Method must save into memory the object at self.model

        Args:
            filename (str):
                name of file to save model to
        """
        # call Keras save function
        return None

    def load(self, filename: str, **kwargs):
        """Method defined by child to load a predictor into memory.

        Loads the object to be assigned to self.model.

        Args:
            filename (str):
                path of file to load
        """
        # call Keras.load function
        model = None
        return model
