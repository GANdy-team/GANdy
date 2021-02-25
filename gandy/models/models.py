"""Parent uncertainty model class implimentation

The structure of GANdy's uncertainty models are defined here. Specific model
type sublass in this module. The models store required shapes, predictors, and
losses for training sessions. They wrap predictors methods in order to accompl-
ish building, training, predicting, and evaluateing.

    Typical usage:
    Not meant to be interacted with directly. Subclasses must define 
    `_build`, `_train`, and `_predict` in order to function properly.
"""
## imports
from typing import Tuple, Iterable, Any, Object, Type

import numpy

import gandy.metrics

## typing
Array = Type[numpy.ndarray]

class NotImplimented(Warning):
    """Warning to indicate that a child class has not yet implimented necessary
    methods. """
    ## pseudocode
    #. define the exception
    pass


class UncertaintyModel:
    """Parent uncertainty model class structure.
    
    Defines the structure for uncertainty models, with method wrappers for eg.
    training, predicting accessing the predictor itself in `self.model`. The 
    `build` method, ran in init, creates the predictor according to the user's
    kwargs and allows for the creation of different complex models eg. GAN vs
    BNN to fit the same parent format. The method also impliments data format
    checking.
    
    Class will raise NotImplimented exception on methods not defined as necess-
    ary in children: `_build`, `_train`, `_predict`
    
    Args:
        xshape (tuple of int): 
            shape of example data, excluding the first dimension
        yshape (tuple of int): 
            shape of target data, excluding the first dimension
        **kwargs:
            keyword arguments to pass to the build method
    """
    
    ## to contain dictionary of callable metric classes from the metrics module
    metrics = {} #gandy.metrics.metric_codex
    """Available metrics defined in gandy.metrics"""
    
    def __init__(self, 
                 xshape: Tuple[int], 
                 yshape: Tuple[int], 
                 **kwargs):
        ## pseudocode
        #. assert inputs
        #. set self shapes
        #. assign self model by running build function
        #. create empty sessions list   
        return
    
    def check(self,
              Xs: Iterable,
              Ys: Iterable = None, 
              **kwargs) -> Tuple[Array]:
        """Attempt to format incoming data.
        
        Assures that a passed set of data has the correct datatypes
        and shapes for the model. Transforms it to numpy if not already.
        
        Args:
            Xs (iterable): examples data to check
            Ys (iterable): label data to check, if present. Default None.
            
        Returns:
            tuple of ndarrays:
                Xs, the formated X data
                Ys, the formated Y data if present
        """
        ## pseudocode
        #. assert data type has shape attribute
        #. check shapes of Xs and Ys against self shapes
        #. raise error if do not match
        #. convert to numpy
        if Ys:
            return Xs, Ys
        else:
            return Xs
    
    def build(self, **kwargs):
        """Construct and store the predictor.
        
        Build a model according to `_build` and assign to `self.model`
        
        Args:
            **kwargs:
                keyword arguments to pass to `_build`
        """
        ## pseudocode
        #. set self model to _build
        return
    
    def _build(self, *args, **kwargs) -> Object:
        """Construct and return the predictor.
        
        Must be implimented in child class. To creates and returns a predictor
        with keyword argument inputs. Raises not implimented warning.
        
        Args:
            *args: 
                arguments defined in child
            **kwargs:
                keyword arguments/hyperparemeters for predictor init.
            
        Raises:
            NotImplimented: 
                warning that child class has not overloaded this method
        Returns:
            None:
                children will return the predictor
        """
        ## pseudocode
        #. raise not implimented 
        #. model is None
        return model
    
    def train(self,
              Xs: Iterable,
              Ys: Iterable, 
              session: str = None, 
              **kwargs):
        """Train the predictor for one session, handled by `_train`.
        
        Trains the stored predictor for a single session according to the
        protocol in `_train`. Stores any returned quantities eg. losses 
        in the `sessions` attribute.
        
        Args:
            Xs (Iterable): 
                Examples data.
            Ys (Iterable): 
                Label data that is targeted for metrics.  
            session (str): 
                Name of training session for storing in losses. default None,
                incriment new name.
            **kwargs: 
                Keyword arguments to pass to `_train` and assign non-default \
                training parameters.
        """
        ## pseudocode
        #. check data inputs with check method - conver to numpy
        #. get metric method
        #. execute _train with formated data and metric (?)
        #. update session losses with session _train losses - maybe create session name
        return 
    
    def _train(self,
               Xs: Array,
               Ys: Array,
               *args,
               **kwargs) -> Any:
        """Train the predictor.
        
        Must be implimented in child class. Trains the stored predictor
        and returns any losses or desired metrics. Up to child to accept metric.
        
        Args:
            Xs (Array): 
                Examples data to train on.
            Ys (Array): 
                Label data that is targeted for metrics for training. 
            *args:
                Positional arguments to be defined by child.
            **kwargs: 
                Keyword arguments to assign non-default training parameters or 
                pass to nested functions.
                
        Returns:
            any: 
                Desired tracking of losses during training. Not implimented
                here, and returns None.
        """
        ## psudocode
        #. raise not implimented
        #. losses is None
        return losses
    
    def predict(self,
                Xs: Iterable, 
                uc_threshold: float = None,
                **kwargs) -> Tuple[Array]:
        """Make predictions on a set of data and return predictions and uncertain-
        ty arrays. 
        
        For a set of incoming data, check it and make predictions with the stored 
        model according to `_predict`. Optionally flag predictions whose uncert-
        ainties excede a desired threshhold
        
        Args:
            Xs (Iterable): 
                Examples data to make predictions of.
            uc_threshold (float): acceptible amount of uncertainty. Predictions of 
                higher ucertainty values will be flagged   
            **kwargs: keyword arguments to pass to `_predict`
                
        Returns:
            tuple of ndarray:
                array of predictions of targets with the same length as Xs
                array of prediction uncertainties of targets withthe same length 
                    as Xs
                (optional) array of flags of uncertain predictions higher than thr-
                    eshhold of same length as Xs
        """
        ## pseudocode
        #. check X data with check function
        #. run _predict to return predictions and uncertainties
        #. if threshhold, return predictions, uncertainties, and flags
        return predictions, uncertainties, flags
    
    def _predict(self, 
                 Xs: Array, 
                 *args, 
                 **kwargs):
        """Make predictions on a set of data and return predictions and uncertain-
        ty arrays.
        
        Must be implimented by child class. Makes predictions on data using
        model at self.model and any other stored objects.
        
        Args:
            Xs (ndarray):
                Example data to make predictions on.
            *args:
                Positional arguments to be defined by child.
            **kwargs: 
                Keyword arguments for predicting.
                
        Returns:
            tuple of ndarray:
                array of predictions of targets with the same length as Xs
                array of prediction uncertainties of targets withthe same length 
                    as Xs
        """
        ## psuedocode
        #. raise not implimented
        #. set pred, unc to None
        return predictions, uncertainties
    
    def score(self, 
              Xs: Iterable,
              Ys: Iterable,
              metric: Union[str, Callable], 
              **kwargs) -> Tuple[float, Array]:
        """Make predictions and score the results according to a defined metric.
        
        For a set of labeled example data, use the the defined `_predict`
        method to make predictions on the data. Then, compare them to the true
        labels according to a desired metric.
        
        Args:
            Xs (Iterable): 
                Examples data to make predictions on.
            Ys (Iterable): 
                Labels of data.
            metric (str): 
                Metric to use, a key in UncertaintyModel.metrics or a metric object
                that takes as input true, predicted, and uncertainty values.
            **kwargs: keyword arguments to pass to `_predict`
            
        Returns:
            float:
                Total score according to metric.
            ndarray:
                Score array for each prediction.
        """
        ## pseudocode
        #. if statement to get metric object from metrics or specified
        #. else raise undefined metric
        #. check data
        #. predictions, uncertainties = execute self.predict on Xs
        #. pass predictions, uncertainties to metric get back costs
        return cost, costs
        
    def save(self, 
             filename: str, 
             format: str = 'h5', 
             **kwargs):
        """Save the model out of memory to the hard drive by specified format.
        
        Args:
            filename (str): 
                path to save model to
            format (str): string name of format to use
                options: TBD
        """
        ## pseudocode
        #. if statements on format
        #   save accordingly
        return
    
    @classmethod
    def load(cls, 
             filename: str, 
             **kwargs):
        """Load a model from hardrive at filename.
        
        Args:
            filename (str): path of file to load
            
        Returns:
            instance of class: the loaded UncertaintyModel
        """
        ## pseudocode
        #. if statements on filename
        #.   create instance of this class
        return instance
    
    @property
    def model(self):
        """predictor: the overall predictor model"""
        return self._model
    
    @model.setter
    def model(self, new_model):
        ## raise exception does not support direct setting, use build function
        return
    
    @model.deleter
    def model(self):
        ## print message about deleting model, build needs to be ran
        return
    
    @property
    def xshape(self):
        """tuple of int: shape of example features"""
        return self._xshape
    
    @xshape.setter(self, new_xshape):
        ## test new shape, delete model
        self._xshape = new_xshape
        return
    
    @property
    def yshape(self):
        """tuple of int: shape of example label"""
        return self._yshape
    
    @yshape.setter(self, new_yshape):
        ## test new shape, delete model
        self._yshape = new_yshape
        return
    
    