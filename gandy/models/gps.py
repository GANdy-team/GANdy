"""Gaussian Process uncertainty models.

Available for classifiers and regressors, use gaussian processes to make predi-
ctions of target values and uncertainties. 

    Typical usage:
    For a set of training data Xs, Ys as arrays with non zeroth dimension
    shapes xshape and yshape: create and train a classifier for one training
    session.
    
    cfr = ucGaussianProcess.C(xshape, yshape)
    cfr.train(Xs, Ys, session='first')
    
    Make predictions on a test set of data Xst, Yst with the same shapes as 
    training:
    predictions, uncertainties = cfr.predict(Xst)
    
    Score the model on the test set using an mse metric:
    score = cfr.evaluate(Xs, Ys, metric='mse')
"""
## imports
from typing import Type, Tuple, Union

import sklearn.gaussian_process
import numpy

import gandy.models.models

## Typing
Model = Type[ucGaussianProcess]
Array = Type[numpy.ndarray]
Predictor = Type[sklearn.gaussian_process]

## The gaussian process uncertainty model
class ucGaussianProcess(gandy.models.models.UncertaintyModel):
    """Gaussian Process Regressor/Classifier Uncertainty Model
    
    Utilizes sklearn's GP objects as an Uncertainty Model, able to make predic-
    tions and uncertainty predictions.
    
    Args:
        xshape (tuple of int): 
            shape of example data, excluding the first dimension
        yshape (tuple of int): 
            shape of target data, excluding the first dimension
        **kwargs:
            keyword arguments to pass to the build method
    """
    def _build(self, 
               model_type: str, 
               **kwargs) -> Predictor:
        """Creates and returns a gaussian process predictor. 
        
        Can be classifier or a regressor, chosen by specifying model type. 
        Accesses scikit-learn for the guassian process predictors.
        
        Args:
            model_type (str):
                Type ('classifier' or 'regressor') of model to assign as the 
                predictor.
            **kwargs: 
                Keyword arguments to pass to `sklearn.gaussian_process-
                .GaussianProcessRegressor/Classifier`
                
        Returns:
            instance of sklearn.gaussian_process: 
                The built predictor.
        """
        ## psueudocode
        #. if statement classifier or regressor
        #    instatiate scikitlearn object with kwargs
        #. else
        #    raise not implimented error
        return model
    
    def _train(self, 
               Xs: Array,
               Ys: Array,
               **kwargs):
        """Trains the gaussian process on training data via covariance kernel.
        
        Trains the predictor accoring to `sklearn.gaussian_process.
        GaussianProcessRegressor/Classifier`. No training losses/metrics 
        associated with the covariance fit, so None is returned.
        
        Args:
            Xs (Array): 
                Examples data to train on.
            Ys (Array): 
                Label data that is targeted for metrics for training. 
            **kwargs:
                Keyword arguments passed to model's fit method.
        Returns:
            None: 
                No losses to return for GP fitting.
        """
        ## pseudocode
        #. fit self model with Xs, Ys
        return None
    
    def _predict(self, 
                 Xs: Array, 
                 **kwargs) -> Tuple[Array]:
        """Make predictions of target on given examples.
        
        Uses the sklearn gaussian process at self.model to make predictions,
        and predictions of uncertainty. Make predictions on unlabeled example
        data.
        
        Args:
            Xs (ndarray):
                Example data to make predictions on.
            **kwargs: 
                keyword arguments passed to predictor's predict method
            
        Returns:
            tuple of ndarray:
                array of predictions of targets with the same length as Xs
                array of prediction uncertainties of targets withthe same length 
                    as Xs
        """
        ## pseudocode
        #. get uncertainties and predictions by passing return_std to 
        #     sklearn object's predict
        return predictions, uncertainties
        
    @classmethod
    def R(cls, *args, **kwargs) -> Model:
        """Alternative to passing model_type as 'regressor' to object 
        initialization.
        
        Arguments:
            *args: 
                positional arguments to pass to init
            **kwargs:
                keyword arguments to pass to init and build methods.
        """
        return cls(*args, model_type = 'regressor', **kwargs)
    
    @classmethod
    def C(cls, *args, **kwargs) -> Model:
        """Alternative to passing model_type as 'classifier' to object 
        initialization.
        
        Arguments:
            *args: 
                positional arguments to pass to init
            **kwargs:
                keyword arguments to pass to init and build methods.
        """
        return cls(*args, model_type = 'classifier', **kwargs)