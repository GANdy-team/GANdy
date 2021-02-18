## imports
import functools
import gandy.metrics


class NotImplimented(Warning):
    '''
    Exception to indicate that a child class has not yet implimented necessary
    methods.
    '''
    ## pseudocode
    #. define the exception
    
    pass


class UncertaintyModel:
    '''
    Parent class for all other models. Children may also inherit other classes.
    
    Defines the structure for uncertainty models, with method wrappers for eg.
    training, predicting accessing the predictor itself in <self.model>. The 
    <build> method, ran in init, creates the predictor according to the user's
    kwargs and allows for the creation of different complex models eg. GAN vs
    BNN to fit the same parent format. The method also impliments data format
    checking.
    
    Class will raise NotImplimented exception on methods not defined as necess-
    ary in children: <_build>, <_train>, <_predict>
    
    '''
    
    ## to contain dictionary of callable metric classes from the metrics module
    metrics = {} #gandy.metrics.metric_codex
    
    def __init__(self, xshape, yshape, **kwargs):
        '''
        Initalize the UncertaintyModel object, including defining shapes and 
        constructing the predictor.
        
        Arguments:
            xshape/yshape - shape of example and target data with unknown count
                type == tuple
                
                eg. for a set of 20 examples with 3 features and 2 targets
                    xshape, yshape = (3,), (2,)
                    
            **kwargs - keyword arguments to be passed to the build method
        '''
        ## pseudocode
        #. assert inputs
        
        #. set self shapes
        
        #. assign self model by running build function
        
        #. create empty losses list
        
        return
    
    
    def check(self, Xs, Ys = None, **kwargs):
        '''
        Assert that a set of data is compatible with model by enforcing that
        the data is of the correct type and shape to the model's parameters.
        Additionally, convert to numpy array if necessary.
        
        Arguments:
            Xs/Ys - examples/targets
                type == array like
        '''
        ## pseudocode
        #. assert data type has shape attribute
        
        #. check shapes of Xs and Ys against self shapes
        
        #. raise error if do not match
        
        #. convert to numpy
        
        return 
    
    def build(self, **kwargs):
        '''
        Construct the model and assign to <self.model>. <_build> Must be implim-
        ented by child.
        
        Arguments:
            **kwargs - keyword arguments to pass to <_build> or pass to nested 
            functions.
        '''
        ## pseudocode
        
        #. set self model to _build
        
        return
    
    def _build(self, **kwargs):
        '''
        Must be implimented in child class. Creates and returns a predictor.
        
        Arguments:
            **kwargs - keyword arguments/hyperparemeters for predictor init.
        '''
        ## pseudocode
        
        #. raise not implimented 
        
        return None
    
    def train(self, Xs, Ys, session, **kwargs):
        '''
        Train the object on a set of data. <_train> must be implimented by 
        child.
        
        Arguments:
            Xs/Ys - training examples/targets
                type == array like
                
            session - name of training session for storing in losses
                type == str
            
            **kwargs - keyword arguments to pass to <_train> and assign non-
                default training parame-ters or pass to nested functions.
        '''
        ## pseudocode
        
        #. check data inputs with check method
        
        #. get metric method
        
        #. execute _train with formated data and metric (?)
        
        #. update session losses with session _train losses
        
        return 
    
    def _train(self, Xs, Ys, **kwargs):
        '''
        Must be implimented in child class. Trains the predictor at self.model
        and returns any losses or desired metrics. Up to child to accept metric.
        
        Arguments:
            Xs/Ys - training examples/targets
                type == ndarray
            
            **kwargs - keyword arguments to assign non-default training parame-
                ters or pass to nested functions.
        '''
        ## psudocode
        
        #. raise not implimented
        
        return losses
    
    def predict(self, Xs, uc_threshold = None, **kwargs):
        '''
        Make predictions on a set of data and return predictions and uncertain-
        ty arrays. According to the defined model. <_predict> must be impliment-
        ed by child.
        
        Arguments:
            Xs - example data to make predictions on
                type == array like
            
            uc_threshold - acceptible amount of uncertainty. Predictions of hi-
                gher ucertainty values will be flagged
                type == float
                
            **kwargs - keyword arguments to pass to <_predict>
                
        Returns:
            predictions - array of predictions of targets with the same length
                as Xs
                type == ndarray
                
            uncertainties - array of prediction uncertainties of targets with 
                the same length as Xs
                type == ndarray
                
            (?) flags - array of flags of uncertain predictions higher than thr-
                eshhold of same length as Xs
                type == ndarray
        '''
        ## pseudocode
        #. check X data with check function
        
        #. run _predict to return predictions and uncertainties
        
        #. if threshhold, return predictions, uncertainties, and flags
        
        return predictions, uncertainties
    
    def _predict(self, Xs, **kwargs):
        '''
        Must be implimented by child class. Makes predictions on data using
        model at self.model.
        
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
        ## psuedocode
        #. raise not implimented
        
        return predictions, uncertainties
    
    def score(self, Xs, metric, **kwargs):
        '''
        Make predictions and score the results according to a defined metric.
        
        Arguments:
            Xs - example data to make predictions on
                type == ndarray
                
            metric - metric to use, a key in UncertaintyModel.metrics or a 
                metric object
                type == str or gandy.metrics.metric
                
            **kwargs - keyword arguments to pass to <predict>
            
        Returns:
            cost - total cost calculated by metric
                type == float
                
            costs - array of costs for X examples
                type == ndarray
        '''
        ## pseudocode
        #. if statement to get metric object from metrics or specified
        #. else raise undefined metric
        
        #. predictions, uncertainties = execute self.predict on Xs
        
        #. pass predictions, uncertainties to metric get back costs
        
        return cost, costs
        
    def save(self, filename, format = 'h5', **kwargs):
        '''
        Save the model out of memory to the hard drive by specified format.
        
        Arguments:
            filename - path to save model to
                type ==str
                
            format - string name of format to use
                type == str
                options:
        '''
        ## pseudocode
        #. if statements on format
        #   save accordingly
        return
    
    @classmethod
    def load(cls, filename, **kwargs):
        '''
        Load a model from hardrive at filename.
        
        Arguments:
            filename - path of 
                type ==str
        '''
        ## pseudocode
        #. if statements on filename
        
        #.   create instance of this class
        
        return instance