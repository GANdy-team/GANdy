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
from typing import Tuple, Iterable, Type, List, Union

import optuna
import numpy

import gandy.models.models

# Typing
Model = Type[gandy.models.models.UncertaintyModel]
Array = Type[numpy.ndarray]
Trial = Type[optuna.trials.Trial]

# class to specify optuna search space from python readable inputs


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
                 param_space: dict,
                 sessions: Union[int, List[str]] = None,
                 k: Union[int, tuple] = None,
                 val_data: Tuple[Array] = None,
                 val_frac: float = None,
                 **kwargs):
        # pseudocode
        # . make sure no overlap of kwargs and param space
        # . store kwargs
        # . make sure only one of k, val_data, val_frac
        # . test input type
        # . set self attributes in proper form
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
        # . hyparams = dict loop self.search_space trial.method(args)
        hyparams = None
        return hyparams

    def _execute_instance(self,
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
        single_loss = None
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
        loss = None
        return loss


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

    def _set_param_space(self, **kwargs):
        """Define the search space from user input search space according to
        gandy.models.hypersearch.SearchableSpace class.

        Not meant to be interacted with directly. Reassigns stored search_space
        if specified.
        """
        # pseudocode
        # . check self.search_space input
        #     raise if None
        # . create empty param_space
        # . for loop self.param_space
        #     param_space = SearchableSpace class
        # set self._param_space
        return

    def _set_objective(self, **kwargs):
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

    def _set_study(self, **kwargs):
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

    @property
    def search_space(self):
        """dict: hyperparameter name to search space parirings"""
        return self._search_space

    @search_space.setter
    def search_space(self, new_search_space):
        # pseudocode
        # . check dict
        self._search_space = new_search_space
        return