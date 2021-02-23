## imports 
import numpy as np
import math 

## Define Dictionary of availiable metrics 
Metric_codex = {}
    

## Parent class for metric 
class Metric:
    ''' 
    
    Implements metric parent class. This class will define the structure of various quality 
    evaluation techniques used for comparing the uncertainty model outputs to real
    experimental data. Children classes will inherent properties of this class.
    
    '''
    
    def __init__(self, predictions, uncertainties = None, real):
        '''
        
        Initializes an instance of the metric class, including the predictions, uncertainties (optional), and real data
        necessary for comparison. 
        
        Arguments:
            predictions - data generated from the uncertainty model 
                
                type == ndarray
            
            uncertainties - optional argument which contains data of uncertainty values generated from the uncertainty model 
                
                type == ndarray
            
            real - data of real values that you want to compare the uncertainty model ouput to (eg. experimental data)  
                
                type == ndarray
        
        '''
        
        ## pseudocode 
        
        #  set self.args
    return 



## Children classes for each relevent metric 
class MSE(Metric):
    
    '''
    
    Mean Squared Error class which defines the structure used for 
    computing the MSE between the passed in datasets. Inherets the
    properties of the parent class Metrics.
    
    '''

    def calculate(**kwargs):
    ''' Method that defines the mathematical formula necessary to compute the MSE. 
        
        Arguments:
        
        **kwargs - necessary keyword arguments to be passed into <calculate> method

        Returns:   

            MSE_value - value of the Mean Squared error computed 

                type == float
     '''

    return MSE_value

class RMSE(Metric):
    '''

    Root Mean Squared Error class which defines the structure used for 
    computing the RMSE between the passed in datasets. Inherets the
    properties of the parent class Metrics.
    
    '''

    def calculate(**kwargs):
    ''' Method that defines the mathematical formula necessary to compute the RMSE. 
        
        Arguments:
        
        **kwargs - necessary keyword arguments to be passed into <calculate> method

        Returns:   

            RMSE_value - value of the Mean Squared error computed 

                type == float
     '''

    return RMSE_value

class F1(Metric):
    '''

    F1 score class which defines the structure used for 
    computing the F1 score between the passed in datasets. Inherets the
    properties of the parent class Metrics.
    
    '''

    def calculate(**kwargs):
    ''' Method that defines the mathematical formula necessary to compute the RMSE. 
        
        Arguments:
        
        **kwargs - necessary keyword arguments to be passed into <calculate> method

        Returns:   

            F1_value - value of the Mean Squared error computed 

                type == float
     '''

    return F1_value