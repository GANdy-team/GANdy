## imports
import gandy.models.models
import sklearn.gaussian_process


## The gaussian process uncertainty model

class ucGaussianProcess(gandy.models.models.UncertaintyModel):
    
    def _build(self, model_type, **kwargs):
        '''
        Creates and returns a gaussian process predictor. Can be classifier or
        a regressor, chosen by specifying model type. Accesses scikit-learn for
        the guassian process predictors.
        
        Arguments:
            model_type - type (classifier or regressor) of model to assign as
                the predictor
                type == str
                
            **kwargs - keyword arguments to pass to <sklearn.gaussian_process-
                .GaussianProcessRegressor/Classifier>
        '''
        ## psueudocode
        #. if statement classifier or regressor
        #    instatiate scikitlearn object with kwargs
        #. else
        #    raise not implimented error
        
        return model
    
    def _train(self, Xs, Ys, **kwargs):
        '''
        Trains the gaussian process on training data via covariance kernel.
        See <sklearn.gaussian_process.GaussianProcessRegressor/Classifier>.
        No training losses/metrics associated with the covariance fit, so
        None is returned.
        
        Arguments:
            Xs/Ys - training examples/targets
                type == ndarray
            
            **kwargs - keyword arguments passed to model's fit method.
        '''
        ## pseudocode
        #. fit self model with Xs, Ys
        
        return None
    
    def _predict(self, Xs, **kwargs):
        '''
        Make predictions of target on examples according to the gaussian 
        process at self.model.
        
        Arguments:
            Xs - example data to make predictions on
                type == ndarray
                
            **kwargs - keyword arguments passed to model's predict method
            
        Returns:
            predictions - predictions of target made for each example, same
                length as Xs
                type == ndarray
            
            uncertainties - uncertainties of predictions according to the
                gaussian process. Same length as Xs
                type == ndarray
        '''
        ## pseudocode
        #. get uncertainties and predictions by passing return_std to 
        #     sklearn object's predict
        
        return predictions, uncertainties
        
    @classmethod
    def R(cls, *args, **kwargs):
        '''
        Alternative to passing model_type as 'regressor' to object initializati-
        on.
        
        Arguments:
            *args - positional arguments to pass to init
        
            **kwargs - keyword arguments to pass to init and build methods.
        '''
        return cls(*args, model_type = 'regressor', **kwargs)
    
    @classmethod
    def C(cls, *args, **kwargs):
        '''
        Alternative to passing model_type as 'classifier' to object initializati-
        on.
        
        Arguments:
            *args - positional arguments to pass to init
        
            **kwargs - keyword arguments to pass to init and build methods.
        '''
        return cls(*args, model_type = 'classifier', **kwargs)