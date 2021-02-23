## imports
import gandy.models.models
import optuna


## class to specify optuna search space from python readable inputs
class SearchableSpace:
    '''
    Wrapper to convert user specified search space into Optuna readable function.
    Attributes:
        func - optuna.trials.Trial method to be used for sampling
            type == bound method in optuna.trials.Trial
            
        name - name of hyperparameter
            type == str
            
        args - positional arguments after name to be passed to func for sampling
            type == tuple
    '''
    
    def __init__(self, hypname, space):
        '''
        Locate and store the correct Optuna search space function and assign
        the necessary arguments.
        
        Arguments:
            hypname - name of hyperparameter
                type == str
                
            space - user defined search space, options below
                list -> catagorical choice
                tuple of ....
                !!!!!! to be defined based on optuna.trial.methods later
        '''
        ## pseudocode
        #. if statement format of space
        #      set self.func, self.args, and self.name
        return
        

## object function class to be optimized
class SubjectObjective:
    '''
    Class to define the objective function in an Optuna study. Not meant to be
    interacted with directly. Supports trial pruning and cross validation.
    '''
    def __init__(self,
                 subject,
                 subject_kwargs,
                 Xs,
                 Ys,
                 search_space,
                 score_kwargs,
                 sessions = 5,
                 k = 5):
        '''
        Initiate the objective for this subject (UncertaintyModel class).
        
        Arguments:
            subject - model class subject to study
                type == child of gandy.models.UncertaintyModel
                
            subject_kwargs - positional and keyword arguments needed to pass to
                subject init in order to create the desired subject. Parameters
                not searched over.
                type == dict
      
            Xs/Ys - examples and targets in the development set used to optimi-
                ze
                type == array like
                
            search_space - dict of hyperparameter search space. Keys are hyper-
                param names and values are instances of SearchableSpace
                type == dict
                
            score_kwargs - keyword arguments to be passed to the score method
                type == dict
                
            sessions - number of training sessions or list of session names 
                to run while checking for pruning
                type == int or list
                
            k - number of folds or indexes in data defining folds to use for
                cross validation
                type == int or list of indexes
                
            
        '''
        ## pseudocode
        #. assert input types
        #. set all self inputs
        #. parse sessions and k based on type
        return
    
    def _sample_params(self, trial):
        '''
        Sample the hyperparameters to be used for this trial. Uses search spaces
        defined at self.search_space (dict of SearchableSpace instances) to ret-
        urn values for this trial.
        
        Agruments:
            trial - Current trial.
                type == optuna.trials.Trial
                
        Returns:
            hyparms - current value of hyperparameters at trial
                type = dict of methods
        '''
        ## pseudocode
        #. hyparams = dict loop self.search_space trial.method(args)
        
        return hyparams
    
    def __call__(self, trial):
        '''
        Function used by optuna to run a single trial. Returns the score to
        minimize.
        
        Agruments:
            trial - Current trial.
                type == optuna.trials.Trial
                
        Returns:
            score - value of the objective to minimize
        '''
        ## pseudocode
        #. sample hypparams for this trial
        
        #. split Xs Ys into folds according to k
        
        #. instantiate k subjects
        
        #. loop for each session
        #     loop for each fold/model
        #         model train method with hyperparams
        #         model score method with score kwargs
        #     session score = average scores
        #     report average score, check if prune
        return score
    

## Hyperparameter search class wrapper for our models and optuna
class OptRoutine:
    '''
    Hyperparameter optimizing routine for uncertainty models. Searches over
    hyperparemeters for a class of uncertainty model for the set producing the
    lowest value of a passed loss metric. Uses a cross validation routine and
    is capable of pruning non-promising models between training sessions. 
    Optimizes objective function of the form 
    gandy.models.hypersearch.SubjectObjective.
    '''
    
    
    def __init__(self, ucmodel_class, 
                 Xs, Ys,
                 subject_kwargs = {}
                 param_space = None,
                 score_kwargs = None,
                 study_kwargs = None,
                 **kwargs):
        '''
        Initialize the routine and define the model class that will be optimi-
        zed.
        
        Arguments:
            ucmodel_class - class to perform the optimization on
                type == child of gandy.models.models.UncertaintyModels
                
            param_space - user defined hyperparameter search space.
                dict of paramname: searchspace. Form of searchspace determines
                the sampling method used. 
                See gandy.models.hypersearch.SearchableSpace for formats
                type == dict
                
            score_kwargs - keyword arguments to pass to the subject's score
                method, such as the metric to use
                type == dict
                
            study_kwargs - keyword arguments to pass to the optuna create_study
                function, such as a pruner to user
                type == dict
                
            subject_kwargs - positional and keyword arguments needed to pass
                to subject init in order to create the desired subject. 
                Parameters not searched over.
                type == dict
                
            **kwargs - additional keyword arguments to pass to construction
        '''
        ## pseudocode 
        #. assert class type is child of uncertainty model
        
        #. set the class to self.subject
        
        #. set self the Xs and Ys data
        
        #. set subject_kwargs here
        
        #. set search space with param space and kwargs
        
        #. set objective with score kwargs and kwargs
        
        #. set study with study kwargs and kwargs
        
        return
    
    def _set_search_space(self, param_space = None, **kwargs):
        '''
        Define the search space from user input param space according to 
        gandy.models.hypersearch.SearchableSpace class. Not meant to be intera-
        cted with directly. Reassigns stored param_space if specified.
        
        Arguments:
            param_space - user defined hyperparameter search space.
                dict of paramname: searchspace. Form of searchspace determines
                the sampling method used. 
                See gandy.models.hypersearch.SearchableSpace for formats
                type == dict
                
        '''
        ## pseudocode
        #. if param_space, set self.param_space
        
        #. check self.param_space input
        #     pass if None
        
        #. create empty search_space
        
        #. for loop self.param_space
        #     search_space = SearchableSpace class
        
        # set self.search_space_
        return 
    
    def _set_objective(self, score_kwargs = None, **kwargs):
        '''
        Define the objective function for optuna to target when optimizing
        hyperparameters. Initates an instance of 
        gandy.models.hypersearch.SubjectObjective, capable of cross validation
        and pruning between sessions. Not meant to be interacted with directly.
        Reassigns stored score_kwargs if specified.
         
        Arguments:
            score_kwargs - keyword arguments to pass to the subject's score
                method, such as the metric to use
                type == dict
                
            **kwargs - keyword arguments to pass to SubjectObjective class,
                such as the number of cross validation folds
        '''
        ## pseudocode
        #. check self.search_space_ exists
        #.   pass if not
        
        #. if score_kwargs, set self.score_kwargs
        
        #. if self.score_kwargs = None, set to {} (empty dict)
        
        #. define SubjectObjective(
        #          self.subject,
#                  self.subject_kwargs,
#                  self.Xs,
#                  self.Ys,
#                  self.search_space_,
#                  self.score_kwargs,
#                  **kwargs)
        
        #. set self.objective_
        return
    
    def _set_study(self, study_kwargs = None, **kwargs):
        '''
        Define the optuna study with create_study. Not meant to be interacted 
        with directly. Reassigns stored study_kwargs if specified.
        
        Arguments:
            study_kwargs - keyword arguments to pass to the optuna create_study
                function, such as a pruner to user
                type == dict
                
            **kwargs - additional keyword arguments
        '''
        ## pseudocode
        #. if study_kwargs, set self.study_kwargs
        
        #. if self.study_kwargs is None, reset to {}

        #. create study optuna.create_study(**self.study_kwargs)
        #. set to self.study_
        return
    
    def optimize(self, 
                 param_space = None, 
                 scoring_kwargs = None, 
                 study_kwargs = None, 
                 **kwargs):
        '''
        Run the optimization study and save the best parameters. Return the
        best model's score.
        
        Arguments:
            (optional if already set)
             param_space - user defined hyperparameter search space.
                dict of paramname: searchspace. Form of searchspace determines
                the sampling method used. 
                See gandy.models.hypersearch.SearchableSpace for formats
                type == dict
                
            score_kwargs - keyword arguments to pass to the subject's score
                method, such as the metric to use
                type == dict
                
            study_kwargs - keyword arguments to pass to the optuna create_study
                function, such as a pruner to user
                type == dict
                
        Returns:
            best_score - best average score of hyperparameters
                type == float
        '''
        ## psuedocode
        #. if param_space specified, set search space
        
        #. if scoring_kwargs specified, set objective
        #. assert objective_ exists
        
        #. if study_kwargs specified, set study
        #. assert study_ exists
        
        #. optimize self.study_ with optimizer kwargs
        
        #. set self.best_params
        
        #. get best_score
        
        return best_score
    
    def train_best(self):
        '''
        Train the subject on the entire dataset with the best found parameters.
        Requires self.optimize to have been executed
        
        Returns:
            best_model - instance of subject with specified static and best 
                searched hyperparameters trained on entire dataset.
        '''
        ## pseudocode
        #. check best_params exist
        
        #. Initiate model with subject_kwargs and best_params
        
        #. train model with best_params training
        
        #. set self.best_model
        
        return best_model
        
        