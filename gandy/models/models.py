"""Parent uncertainty model class implimentation

The structure of GANdy's uncertainty models are defined here. Specific model
type sublass in this module. The models store required shapes, predictors, and
losses for training sessions. They wrap predictors methods in order to accompl-
ish building, training, predicting, and evaluateing.

    Typical usage:
    Not meant to be interacted with directly. Subclasses must define
    `_build`, `_train`, and `_predict` in order to function properly.
"""

# imports
import time
from typing import Tuple, Iterable, Any, Type, Callable, Union

import numpy

import gandy.quality_est.metrics

# typing
Array = Type[numpy.ndarray]


class NotImplimented(Exception):
    """Warning to indicate that a child class has not yet implimented necessary
    methods.

    Args:
        inst - the class instance that raises this exception
    """

    def __init__(self, inst):
        self.message = """This method has not yet been implimented by
 this class: `{}`.""".format(inst.__class__)
        super().__init__(self.message)
        return


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

    Attributes:
        sessions (list of tuple):
            Stored losses from training sessions. When train is called, a new
            tuple is appended of (session name, losses) where losses is
            determined by the output of _train.
    """
    metrics = gandy.quality_est.metrics
    """Available metrics defined in gandy.metrics"""

    def __init__(self,
                 xshape: Tuple[int],
                 yshape: Tuple[int],
                 **kwargs):
        self._model = None
        self.xshape = xshape
        self.yshape = yshape
        self.sessions = {}
        self.build(**kwargs)
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
        if hasattr(Xs, 'shape'):
            pass
        else:
            raise AttributeError('Xs has no shape attribute, ensure the\
 passed data has a shape')
        if Ys is not None:
            if hasattr(Ys, 'shape'):
                pass
            else:
                raise AttributeError('Ys has no shape attribute, ensure the\
 passed data has a shape')

        try:
            Xs_ = numpy.array(Xs).astype(numpy.float64)
        except ValueError:
            raise TypeError('X data contains non numerics.')

        try:
            Xs_ = Xs_.reshape(-1, *self.xshape)
            if len(Xs_) != len(Xs):
                raise ValueError()
        except ValueError:
            raise ValueError('Cannot reshape X data ({}) to the model input\
 shape ({}). ensure the correct shape of data'.format(
                Xs.shape[1:], self.xshape)
            )
        if Ys is not None:
            Ys_ = numpy.array(Ys)
            try:
                Ys_ = Ys_.reshape(-1, *self.yshape)
                if len(Ys_) != len(Ys):
                    raise ValueError()
            except ValueError:
                raise ValueError('Cannot reshape Y data ({}) to the model\
 input shape ({}). ensure the correct shape of data'.format(
                    Ys.shape[1:], self.yshape)
                )
            if len(Xs_) == len(Ys_):
                pass
            else:
                raise ValueError('X and Y data do not have the same number of\
 examples. Ensure the data are example pairs.')
            return Xs_, Ys_
        else:
            return Xs_

    def build(self, **kwargs):
        """Construct and store the predictor.

        Build a model according to `_build` and assign to `self.model`

        Args:
            **kwargs:
                keyword arguments to pass to `_build`
        """
        self._model = self._build(**kwargs)
        return

    def _build(self, *args, **kwargs) -> Callable:
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
        raise NotImplimented(self)
        model = None
        return model

    def train(self,
              Xs: Iterable,
              Ys: Iterable,
              metric: Union[str, Callable] = None,
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
                use clock time.
            metric (str):
                Metric to use, a key in UncertaintyModel.metrics or a metric
                objectthat takes as input true, predicted, and uncertainty
                values.
            **kwargs:
                Keyword arguments to pass to `_train` and assign non-default \
                training parameters.
        """
        if session is not None:
            sname = session
        else:
            sname = 'Starttime: ' + str(time.clock())
        metric = self._get_metric(metric)

        Xs_, Ys_ = self.check(Xs, Ys)
        losses = self._train(Xs_, Ys_, metric=metric, **kwargs)
        self.sessions[sname] = losses
        return

    def _train(self,
               Xs: Array,
               Ys: Array,
               *args,
               metric: Callable = None,
               **kwargs) -> Any:
        """Train the predictor.

        Must be implimented in child class. Trains the stored predictor
        and returns any losses or desired metrics. Up to child to accept
        metric.

        Args:
            Xs (Array):
                Examples data to train on.
            Ys (Array):
                Label data that is targeted for metrics for training.
            metric (callable):
                Metric to use, takes true, predicted, uncertainties to
                compute a score.
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
        raise NotImplimented(self)
        losses = None
        return losses

    def predict(self,
                Xs: Iterable,
                uc_threshold: float = None,
                **kwargs) -> Tuple[Array]:
        """Make predictions on a set of data and return predictions and
        uncertainty arrays.

        For a set of incoming data, check it and make predictions with the
        stored model according to `_predict`. Optionally flag predictions whose
        uncertainties excede a desired threshhold.

        Args:
            Xs (Iterable):
                Examples data to make predictions of.
            uc_threshold (float): acceptible amount of uncertainty.
                Predictions of higher ucertainty values will be flagged
            **kwargs: keyword arguments to pass to `_predict`

        Returns:
            tuple of ndarray:
                array of predictions of targets with the same length as Xs
                array of prediction uncertainties of targets withthe same
                    length as Xs
                (optional) array of flags of uncertain predictions higher
                    than threshhold of same length as Xs
        """
        Xs_ = self.check(Xs)
        if uc_threshold is not None:
            try:
                thresh = numpy.float64(uc_threshold)
            except ValueError:
                raise TypeError(
                    'The threshold ({}) cannot be made a float.'.format(
                        uc_threshold)
                )
        else:
            pass

        predictions, uncertainties = self._predict(Xs_, **kwargs)
        predictions = numpy.array(predictions).reshape(len(Xs), *self.yshape)
        uncertainties = numpy.array(uncertainties).reshape(
            len(Xs), *self.yshape)
        try:
            uncertainties = uncertainties.astype(numpy.float64)
        except ValueError:
            raise TypeError('Uncertainties are not numeric. Check the return\
 of the _predict method.')

        if uc_threshold is not None:
            flags = uncertainties > thresh
            return predictions, uncertainties, flags
        else:
            return predictions, uncertainties

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
                array of prediction uncertainties of targets withthe same
                    length as Xs
        """
        raise NotImplimented(self)
        predictions, uncertainties = None, None
        return predictions, uncertainties

    def _get_metric(self, metric_in: Union[None, Callable, str]):
        """Accesses gandy metrics to retrieve the correct metric depending on
        input

        Args:
            metric_in (str, callable, None):
                metric name to get or callable to use"""
        # if statement, None, string, callable
        if metric_in is None:
            metric_out = None
        elif callable(metric_in):
            metric_out = metric_in
        elif isinstance(metric_in, str):
            if hasattr(self.metrics, metric_in):
                metric_out = getattr(self.metrics, metric_in)
            else:
                raise AttributeError('gandy has no metric called {}'.format(
                    metric_in)
                )
        else:
            raise ValueError('Unable to parse metric')
        return metric_out

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
                Metric to use, a key in UncertaintyModel.metrics or a metric
                object that takes as input true, predicted, and uncertainty
                values.
            **kwargs: keyword arguments to pass to `_predict`

        Returns:
            float:
                Total score according to metric.
            ndarray:
                Score array for each prediction.
        """
        metric = self._get_metric(metric)
        Xs_, Ys_ = self.check(Xs, Ys)

        predictions, uncertainties = self.predict(Xs_, **kwargs)

        metric_value, metric_values = metric(Ys_, predictions, uncertainties)
        metric_values = numpy.array(metric_values).astype(numpy.float64)
        metric_values = metric_values.reshape(len(Xs), -1)

        return metric_value, metric_values

    def save(self,
             filename: str,
             **kwargs):
        """Save the model out of memory to the hard drive. Must be overloaded
        by child.

        Args:
            filename (str):
                path to save model to, no extension
        """
        raise NotImplimented(self)
        return

    @classmethod
    def load(cls,
             filename: str,
             **kwargs):
        """Load a model from hardrive at filename.

        Must be overloaded by child.

        Args:
            filename (str):
                path of file to load

        Returns:
            instance of class: the loaded UncertaintyModel
        """
        raise NotImplimented(cls)
        instance = None
        return instance

    @property
    def model(self):
        """predictor: the overall predictor model"""
        return self._model

    @model.setter
    def model(self, new_model):
        # raise exception does not support direct setting, use build function
        raise RuntimeError(
            'Do not set the model directly, execute the build method')
        return

    @model.deleter
    def model(self):
        # print message about deleting model, build needs to be ran
        if self._model is not None:
            print('WARNING: model no longer valid, deleting. Rerun build()')
        self._model = None
        return

    @property
    def xshape(self):
        """tuple of int: shape of example features"""
        return self._xshape

    @xshape.setter
    def xshape(self, new_xshape):
        # test new shape, delete model
        if isinstance(new_xshape, tuple):
            if all([isinstance(dim, int) for dim in new_xshape]):
                pass
            else:
                raise TypeError('Non-int dimension found in xshape input')
        else:
            raise TypeError('xshape must be a tuple (dims of an x datum)')
        self._xshape = new_xshape
        del self.model
        return

    @property
    def yshape(self):
        """tuple of int: shape of example label"""
        return self._yshape

    @yshape.setter
    def yshape(self, new_yshape):
        # test new shape, delete model
        if isinstance(new_yshape, tuple):
            if all([isinstance(dim, int) for dim in new_yshape]):
                pass
            else:
                raise TypeError('Non-int dimension found in yshape input')
        else:
            raise TypeError('yshape must be a tuple (dims of a y datum)')
        self._yshape = new_yshape
        del self.model
        return
