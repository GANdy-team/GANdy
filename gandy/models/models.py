## imports




class UncertaintyModel:
    '''
    
    '''
    
    def __init__(self, xshape, yshape, **kwargs):
        '''
        Initalize the model object
        '''
        return
    
    
    def check(self, Xs, Ys, **kwargs):
        '''
        Assert that a set of data is okay with the model shapes
        '''
        return
    
    def build(self):
        '''
        Construct the model
        '''
        
        return
    
    def train(self, Xs, Ys, **kwargs):
        '''
        Train the object on a set of data
        '''
        return
    
    def predict(self, Xs, uc_threshold = None, **kwargs):
        '''
        Make predictions on a set of data and return predictions and uncertainty
        '''
        return predictions, uncertainty
    
    def save(self, format = 'h5', **kwargs):
        '''
        Save the model out of memory to the hard drive
        '''
        return
    
    def load(self, filename, **kwargs):
        '''
        Load a model from hardrive
        '''
        return
        