"""
Bayes NN.

This contains the Bayes NN class, based on the KEras tutorial at
https://keras.io/examples/keras_recipes/bayesian_neural_networks/
"""
# typing imports
import inspect
from typing import Any, Callable, Type, Union, Tuple

# 3rd part imports
import numpy
import tensorflow as tf
import tensorflow_probability as tfp

# gandy
import gandy.models.models

# typing
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
    def __init__(self,
                 xshape: Tuple[int],
                 yshape: Tuple[int],
                 train_size: int,
                 **kwargs):
        super().__init__(xshape, yshape, train_size=train_size, **kwargs)
        return

    def prior(self, kernel_size, bias_size, dtype=None) -> Callable:
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
        try:
            kernel_size = int(kernel_size)
            bias_size = int(bias_size)
        except BaseException:
            raise TypeError('Cannot convert kernel and bias to int.')
        n = kernel_size + bias_size
        prior_model = tf.keras.Sequential(
            [
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.MultivariateNormalDiag(
                        loc=tf.zeros(n), scale_diag=tf.ones(n)
                    )
                )
            ]
        )
        return prior_model

    # Define variational posterior weight distribution as multivariate
    # Gaussian. Note that the learnable parameters for this
    # distribution are the means, variances, and covariances.
    def posterior(self, kernel_size, bias_size, dtype=None) -> Callable:
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
        try:
            kernel_size = int(kernel_size)
            bias_size = int(bias_size)
        except BaseException:
            raise TypeError('Cannot convert kernel and bias to int.')
        n = kernel_size + bias_size
        posterior_model = tf.keras.Sequential(
            [
                tfp.layers.VariableLayer(
                    tfp.layers.MultivariateNormalTriL.params_size(n),
                    dtype=dtype
                ),
                tfp.layers.MultivariateNormalTriL(n)
            ]
        )
        return posterior_model

    # Since the output of the model is a distribution, rather than a
    # point estimate, we use the negative loglikelihood as our loss function
    # to compute how likely to see the true data (targets) from the
    # estimated distribution produced by the model.
    def negative_loglikelihood(self, targets, estimated_distribution) -> Array:
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
        try:
            nll = estimated_distribution.log_prob(targets)
        except AttributeError:
            raise AttributeError('Passed distribution does not have the\
 log_prob method')
        return -nll

    # overridden method from UncertaintyModel class
    def _build(self,
               train_size: int,
               task_type: str = 'regression',
               activation: Union[Callable, str] = 'sigmoid',
               optimizer: Union[Callable, str] = 'Adam',
               neurons: Tuple[int] = (12, 12, 12),
               metrics=['MSE'],
               **kwargs) -> Callable:
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
        # parse kwargs
        layer_kwargs = {}
        optimizer_kwargs = {}
        output_kwargs = {}

        for k, v in kwargs.items():
            if k.startswith('optimizer_'):
                optimizer_kwargs[k[10:]] = v
            elif k.startswith('layer_'):
                layer_kwargs[k[6:]] = v
            elif k.startswith('output_'):
                output_kwargs[k[7:]] = v
            else:
                print(k + ' is not a valid hyperparamter, ignoring')
                pass

        inputs = tf.keras.Input(self.xshape)
        f = tf.keras.layers.BatchNormalization()(inputs)

        # loop through each neuron
        for n in neurons:
            f = tfp.layers.DenseVariational(
                units=n,
                make_prior_fn=self.prior,
                make_posterior_fn=self.posterior,
                kl_weight=1 / train_size,
                activation=activation,
                **layer_kwargs
            )(f)

        # determine output type
        if task_type == 'regression':
            outl = tfp.layers.IndependentNormal
            distr = tf.keras.layers.Dense(
                outl.params_size(self.yshape),
                **output_kwargs
            )(f)
            outputs = outl(self.yshape)(distr)
        elif task_type == 'classification':
            outl = tfp.layers.CategoricalMixtureOfOneHotCategorical
            distr = tf.keras.layers.Dense(
                outl.params_size(self.yshape, 2),
                **output_kwargs
            )(f)
            outputs = outl(self.yshape, 2)(distr)
        else:
            raise ValueError('Unknown task typle {}'.format(task_type))

        if not callable(optimizer):
            if isinstance(optimizer, str):
                optimizer = getattr(tf.keras.optimizers, optimizer)
                optimizer = optimizer(**optimizer_kwargs)
            else:
                pass
        else:
            optimizer = optimizer(**optimizer_kwargs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer,
                      loss=self.negative_loglikelihood,
                      metrics=metrics)

        return model

    # overridden method from UncertaintyModel class
    def _train(self,
               Xs: Array,
               Ys: Array,
               metric: Callable = None,
               **kwargs) -> Any:
        '''
        Trains BNN model on data

        Arguments:
            Xs/Ys - training examples/targets
                type == ndarray

            **kwargs - keyword arguments to assign non-default training parame-
                ters or pass to nested functions.
        '''
        # filter out kwargs
        fc_kwargs = {key: val for (key, val) in kwargs.items()
                     if key in inspect.getfullargspec(self.model.fit)[0]}
        print(inspect.getfullargspec(self.model.fit)[0])

        losses = self.model.fit(Xs, Ys, **fc_kwargs)
        return losses

    # overridden method from UncertaintyModel class
    def _predict(self,
                 Xs: Array,
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
        dists = self.model(Xs)
        predictions = dists.mean().numpy()
        uncertainties = dists.stddev().numpy()
        return predictions, uncertainties

    def save(self, filename: str, **kwargs):
        """Method defined by child to save the predictor.

        Method must save into memory the object at self.model

        Args:
            filename (str):
                name of file to save model to
        """
        # call Keras save function
        self.model.save(filename)
        return None

    @classmethod
    def load(cls, filename: str, **kwargs):
        """Method defined by child to load a predictor into memory.

        Loads the object to be assigned to self.model.

        Args:
            filename (str):
                path of file to load
        """
        model = tf.keras.models.load_model(filename)
        xshape = model.input_shape[1:]
        yshape = model.layers[-1].get_config()['event_shape']
        inst = cls.__new__(cls)
        inst._xshape = xshape
        inst._yshape = yshape
        inst._model = model
        return inst
