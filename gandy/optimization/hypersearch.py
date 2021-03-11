"""Optimization routine to find the best hyperparameter values for an
UncertaintyModel.

Impliment's Optuna's hyperparameter studies in order to probe hyperparameter
values. An UncertaintyModel class is chosen as as subject for the study, and
an objectcive function returning a cost quantity to minimize is defined that
incorporates the studied subject and the chosen hyperparameters search spaces


    Typical usage:

    Define `search_space` a dictionary of hyperparameters and their respective
    space to search.

    search_space = {'hyp1': [choice1, choice2, choice3],
                    'hyp2': (low, high, uniform)}

    For a set of development data Xs, Ys as arrays with non zeroth dimension
    shapes xshape and yshape:

    opt = OptRoutine(UncertaintyModel, Xs, Ys, search_space, xshape=xshape,
                     yshape=yshape)

    Optimize with 3 fold CV
    opt.optimize(k=3)
    best_params = opt.best_params # dict of the values in search_space for each
                                  # hyp found to be best
"""

# imports
from typing import Tuple, Iterable, Type, List, Union, Callable

import numpy
import optuna.trial
import optuna
import sklearn.model_selection

import gandy.models.models

# Typing
Model = Type[gandy.models.models.UncertaintyModel]
Array = Type[numpy.ndarray]
Trial = Type[optuna.trial.Trial]


class SearchableSpace:
    """Wrapper to convert user specified search space into Optuna readable
    function.

    Args:
        hypname (str):
            Name of hyperparameter.
        space (tuple or list):
            The user defined hyperparameter space to search, determined by form
            Options - TBD

    Attributes:
        func (optuna.trials.Trial method):
            Function to be used for sampling of hyperparams.
        hypname (str):
            Name of hyperparameter.
        args (tuple): Positional arguments after name to be passed to func for
            sampling
    """

    def __init__(self, hypname, space):
        # pseudocode
        # . if statement format of space
        #      set self.func, self.args, and self.hypname
        self.name = hypname

        # categorical
        if isinstance(space, list):
            self.args = (space,)
            self.func = optuna.trial.Trial.suggest_categorical
        # others
        elif isinstance(space, tuple):
            # check if we need to add a parameter to the end (len =2)
            if len(space) == 2:
                if all(isinstance(i, int) for i in space):
                    space_ = list(space)
                    space_.append(1)
                    space_ = tuple(space_)
                    print(
                        'Assuming uniform integer sampling for hyperparameter\
 {} with search space specified as Tuple[int] with len 2'.format(hypname)
                    )
                elif all(isinstance(i, float) for i in space):
                    space_ = list(space)
                    space_.append('uniform')
                    space_ = tuple(space_)
                    print(
                        'Assuming uniform continuous sampling for\
 hyperparameter {} with search space specified as Tuple[float] with\
  len 2'.format(hypname)
                    )
                else:
                    raise ValueError('hyperparameter space as tuple must have\
 the first two arguments be both float or integer')

            elif len(space) == 3:
                space_ = space
            else:
                raise ValueError(
                    'space as a tuple indicates (min, max, step/type) and\
 should have 2 or 3 contents, not {}'.format(len(space)))

            if not isinstance(space_[0], type(space_[1])):
                raise ValueError('hyperparameter space as tuple must have\
 the first two arguments be both float or integer')
            # integer choice
            elif isinstance(space_[0], int):
                if not isinstance(space_[2], int):
                    raise ValueError('First two values in space are int,\
 indicating integer selection, but the third (step size) is not an int')
                else:
                    pass
                self.args = space_
                self.func = optuna.trial.Trial.suggest_int
            elif isinstance(space_[0], float):
                if space_[2] == 'uniform':
                    self.args = space_[:2]
                    self.func = optuna.trial.Trial.suggest_uniform
                elif space_[2] == 'loguniform':
                    self.args = space_[:2]
                    self.func = optuna.trial.Trial.suggest_loguniform
                elif isinstance(space_[2], float):
                    self.args = space_
                    self.func = optuna.trial.Trial.suggest_discrete_uniform
                else:
                    raise ValueError(
                        'Unknown specification for float suggestion {}, should\
 be "uniform" or "loguniform" indicating the distribution, or a float,\
  indicating a discrete spep'
                    )

            else:
                raise ValueError('hyperparameter space as tuple must have\
 the first two arguments be both float or integer')

        else:
            raise TypeError(
                'space must be a list or tuple, not {}'.format(type(space))
            )
        return


# object function class to be optimized
class SubjectObjective:
    """Objective function definition in an Optuna study.

    Not meant to be interacted with directly. Supports trial pruning and cross
    validation. Define an objective function for hyperparameter optimization
    considering a subject class.

    Args:
        subject (UncertaintyModel):
            A class of UncertaintyModel, the subject of the study
        Xs (ndarray):
            Examples feature data in the development set to use for study.
        Ys (ndarray):
            Examples label data in the development set to use for study.
        param_space (dict):
            Pairs of {hypname: sample_function} where sample_function is a
            optuna Trial method used for sampleing
        sessions (int or list of str):
            Number of training sessions to execute per model or names of
            sessions, checking for pruning if necessary
        k (int or):
            Number of folds for cross validation or tuple of fold indexes in
            data. Default None, don't use cross validation.
        val_data (tuple of array):
            Validation (examples, targets) to use for scoring. Default None,
            use Xs, Ys data.
        val_frac (float):
            Fraction of passed data to use for scoring. Default None, don't
            split data.
        **kwargs:
            Keyword arguments to pass to constructor, training, and scoring.
    """

    def __init__(self,
                 subject: Model,
                 Xs: Array,
                 Ys: Array,
                 param_space: list,
                 sessions: Union[int, List[str]] = None,
                 k: Union[int, tuple] = None,
                 val_data: Iterable[Array] = None,
                 val_frac: float = None,
                 **kwargs):
        # pseudocode
        # . make sure no overlap of kwargs and param space
        # . store kwargs
        # . make sure only one of k, val_data, val_frac
        # . test input type
        # . set self attributes in proper form
        for param in param_space:
            if param.name in kwargs.keys():
                raise ValueError(
                    'Whoa! A searchable parameter {} is also passed as a\
 keyword argument. A parameter cannot be both searched and stationary.\
 '.format(param.name)
                )
            else:
                pass

        self.kwargs = kwargs
        self.param_space = param_space
        self.subject = subject

        # store data
        if len(Xs) != len(Ys):
            raise ValueError('Data should be the same length')
        self.Xs = Xs
        self.Ys = Ys

        # check only one argument passed
        passed = [i is not None for i in [k, val_data, val_frac]]
        if numpy.sum(passed) > 1:
            raise ValueError('Only one k, val_data, val_frac acceptable')
        elif numpy.sum(passed) == 0:
            val_frac = 0.2
        else:
            pass

        if k is not None:
            self.k = k
        else:
            self._k = None
        if val_data is not None:
            self.val_data = val_data
        else:
            self._val_data = None
        if val_frac is not None:
            self.val_frac = val_frac
        else:
            self._val_frac = None

        if isinstance(sessions, int):
            self.sessions = range(sessions)
        elif isinstance(sessions, list):
            self.sessions = sessions
        elif sessions is None:
            self.sessions = range(1)
        else:
            raise TypeError(
                'sessions should be a list of names or an integer number,\
 not {}'.format(type(sessions))
            )

        return

    def _sample_params(self, trial: Trial) -> dict:
        """Sample the hyperparameters to be used for this trial.

        Uses space defined at self.param_space (dict of SearchableSpace
        instances) to return values for this trial.

        Args:
            trial (optuna Trial):
                Current trial.

        Returns:
            hyparms (dict):
                Mapping of hyperparameter values to use for this trial
        """
        # pseudocode
        # . hyparams = dict loop self.param_space trial.method(args)
        hyparams = {}
        for param in self.param_space:
            print(param.func)
            hyparams[param.name] = param.func(trial, param.name, *param.args)
        print(hyparams)
        return hyparams

    def _execute_instance(self,
                          instance: Callable,
                          hyparams: dict,
                          train_data: Tuple[Array],
                          val_data: Tuple[Array]) -> float:
        """Train and score on validation data a single subject instance.

        Args:
            hyparms (dict):
                Mapping of hyperparameter values to use for this trial
            train_data (tuple of ndarray):
                Training (examples, targets) to use for scoring.
            val_data (tuple of ndarray)
                Validation (examples, targets) to use for scoring.

        Returns:
            float:
                The score of the model on the validation data.
        """
        # pseudocode
        # . construct model with hyparms and self kwargs
        # . train model with hyparms and self kwargs
        # . score model with self kwargs
        instance.train(*train_data, **self.kwargs, **hyparams)
        single_loss = instance.score(*val_data, **self.kwargs, **hyparams)
        return single_loss

    def __call__(self, trial: Trial) -> float:
        """Function used by optuna to run a single trial. Returns the score to
        minimize.

        Args:
            trial (optuna Trial):
                Current trial.

        Returns:
            float:
                The score of this trial, the quantity to minimize.
        """
        # pseudocode
        # . sample hypparams for this trial
        # . depending on val_data, k, or val_frac
        # .    split data based on above
        # .    for each session, execute instances
        # .    check for prune
        hyparams = self._sample_params(trial)

        if self.k is not None:
            # need k instances
            instances = [
                self.subject(**self.kwargs, **hyparams) for i in self.k
            ]
            for session in self.sessions:
                losses = []
                for fold, instance in enumerate(instances):
                    train_data = (self.Xs[self.k[fold][0]],
                                  self.Ys[self.k[fold][0]])
                    val_data = (self.Xs[self.k[fold][1]],
                                self.Ys[self.k[fold][1]])
                    single_loss = self._execute_instance(instance,
                                                         hyparams,
                                                         train_data,
                                                         val_data)
                    losses.append(single_loss)
                loss = numpy.mean(losses)

                trial.report(loss, session)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return loss

        if self.val_data is not None:
            instance = self.subject(**self.kwargs, **hyparams)
            for session in self.sessions:
                loss = self._execute_instance(instance,
                                              hyparams,
                                              (self.Xs, self.Ys),
                                              self.val_data)
                trial.report(loss, session)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return loss

        if self.val_frac is not None:
            instance = self.subject(**self.kwargs, **hyparams)
            Xt, Xv, Yt, Yv = sklearn.model_selection.train_test_split(
                self.Xs, self.Ys, test_size=self.val_frac
            )
            train_data = (Xt, Yt)
            val_data = (Xv, Yv)
            for session in self.sessions:
                loss = self._execute_instance(instance,
                                              hyparams,
                                              train_data,
                                              val_data)
                trial.report(loss, session)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return loss

    @property
    def k(self):
        """list of fold indicies for incoming data, (train ind, val ind)"""
        return self._k

    @k.setter
    def k(self, new_k):
        # if int convert to indexes
        # otherwise check proper form
        if not isinstance(new_k, int):
            for inds in new_k:
                if len(inds) != 2:
                    raise ValueError(
                        'k if not an integer of no. folds, should be an\
 iterable of fold idexes (train ind, test ind). {} does not have the correct\
  format of Iterable of len(2) iterables'.format(new_k)
                    )
        else:
            kfold = sklearn.model_selection.KFold(new_k, shuffle=True)
            new_k = kfold.split(self.Xs)
            new_k = list(new_k)
        self._k = new_k
        del self.val_data
        del self.val_frac
        return

    @k.deleter
    def k(self):
        self._k = None
        # print message
        print('Cannot have more than one k, val_data, val_frac. Deleting k')
        return

    @property
    def val_data(self):
        """tuple of ndarray, (train data, test data)"""
        return self._val_data

    @val_data.setter
    def val_data(self, new_val_data):
        # check tuple of array
        if len(new_val_data) != 2:
            raise ValueError(
                'val_data should be an iterable of len 2, x and y data arrays'
            )
        else:
            for d in new_val_data:
                if not hasattr(d, "__len__"):
                    raise ValueError('One object passed in val_data not iter')
            if len(new_val_data[0]) != len(new_val_data[1]):
                raise ValueError('val_data x, y different lengths')
        self._val_data = new_val_data
        del self.k
        del self.val_frac
        return

    @val_data.deleter
    def val_data(self):
        self._val_data = None
        # print message
        print('Cannot have more than one k, val_data, val_frac. \
 Deleting val_data')
        return

    @property
    def val_frac(self):
        """fraction of incoming data to use as validation data,
        randomly sample"""
        return self._val_frac

    @val_frac.setter
    def val_frac(self, new_val_frac):
        # check float
        try:
            new_val_frac = float(new_val_frac)
        except BaseException:
            raise TypeError('Cannot convert {} to float'.format(new_val_frac))
        if not 0.0 < new_val_frac < 1.0:
            raise ValueError('val_frac must be between 0 and 1')
        self._val_frac = new_val_frac
        del self.k
        del self.val_data
        return

    @val_frac.deleter
    def val_frac(self):
        self._val_frac = None
        print('Cannot have more than one k, val_data, val_frac.\
 Deleting val_frac')
        return


# Hyperparameter search class wrapper for our models and optuna
class OptRoutine:
    """Hyperparameter optimizing routine for uncertainty models.

    Searches over hyperparemeters for a class of uncertainty model for the set
    producing thelowest value of a passed loss metric. Uses a cross validation
    routine and is capable of pruning non-promising models between training
    sessions. Optimizes objective function of the form
    `gandy.models.hypersearch.SubjectObjective`.

    Args:
        subject (UncertaintyModel):
            A class of UncertaintyModel, the subject of the study
        Xs (Iterable):
            Examples feature data in the development set to use for study.
        Ys (Iterable):
            Examples label data in the development set to use for study.
        search_space (dict):
            Mapping of the hyperparameters to search over as {name: space}
            where space represents a search space based on its format.
            Options - TBD
    """

    def __init__(self,
                 subject: Model,
                 Xs: Iterable,
                 Ys: Iterable,
                 search_space=None,
                 **kwargs):
        # pseudocode
        # . assert class type is child of uncertainty model
        # . set the class to self.subject
        # . set self the Xs and Ys data after taking values
        # . save all_kwargs
        return

    def _set_param_space(self):
        """Define the search space from user input search space according to
        gandy.models.hypersearch.SearchableSpace class.

        Not meant to be interacted with directly. Reassigns stored search_space
        if specified.
        """
        # # pseudocode
        # . create empty param_space
        # . for loop self.param_space
        #     param_space = SearchableSpace class
        # set self._param_space
        return

    def _set_objective(self):
        """Define the objective function for optuna to target when optimizing
        hyperparameters.

        Initates an instance of gandy.models.hypersearch.SubjectObjective,
        capable of cross validation and pruning between sessions. Not meant to
        be interacted with directly.

        Args:
            **kwargs:
                Keyword arguments to pass to SubjectObjective class,
                such as the number of cross validation folds or constructor
                kwargs
        """
        # pseudocode
        # . check self._param_space exists
        # .   try to set if not
        # . define SubjectObjective(
        #          self.subject,
#                  self.Xs,
#                  self.Ys,
#                  **kwargs)

        # . set self.objective
        return

    def _set_study(self):
        """Define the optuna study with create_study.

        Not meant to be interacted with directly. Creates the study to be used

        Args:
            **kwargs:
                Keyword arguments for optuna create study.
        """
        # pseudocode
        # . create study optuna.create_study
        # . set to self.study
        return

    def optimize(self,
                 search_space: dict = None,
                 **kwargs) -> float:
        """Run the optimization study and save the best parameters. Return the
        best model's score.

        Args:
            search_space (dict):
                Mapping of the hyperparameters to search over as {name: space}
                where space represents a search space based on its format.
                Options - TBD
            **kwargs:
                Keyword arguments to pass to constructor, optimizer, etc.

        Returns:
            best_score (float):
                best score of all hyperparameters searched
        """
        # psuedocode
        # . if search_space specified set
        # . update all kwargs with these
        # . set optimizer, study with all kwargs
        # . set self.best_params
        # . get best_score
        best_score = None
        return best_score

    def train_best(self, **kwargs) -> Model:
        """Train the subject on the entire dataset with the best found
        parameters.

        Requires self.optimize to have been executed or best_params to have
        been specified.

        Args:
            **kwargs:
                Keyword arguments to pass to the constructor and trainer

        Returns:
            best_model (UncertaintyModel):
                Instance of subject with specified static and best searched
                hyperparameters trained on entire dataset.
        """
        # pseudocode
        # . check best_params exist
        # . update all kwargs
        # . Initiate model with  and best_params and kwargs
        # . train model with best_params training and kwargs
        # . set self.best_model
        best_model = None
        return best_model
