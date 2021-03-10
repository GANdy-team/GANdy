'''
Metrics module: contains some relevent metrics to assess the performance of
machine learning models.

This module implements a parent metric class that contains necessary
initialization arguments and automatically calls a calculate method to
compute a given metric. Required intial arguments include the machine
learning model output predictions and real values for comparison. Optionally,
the user may input uncertainties if a given model outputs them. The
properties of the parent class are then inhereted by individual children
classes which define the exact mathematical operations to compute a specific
metric. Calling a specific metric child class will compute a given metric and
return the total value and/or individual values of that metric based on the
input data provided.
'''

# Imports
from typing import Type, Tuple

import numpy as np

# Typing
Array = Type[np.ndarray]


# Parent class for metric
class Metric:
    '''
    Implements metric parent class. This class will define the structure of
    various quality evaluation techniques used for comparing the uncertainty
    model outputs to real experimental data. Children classes will inherent
    properties of this class.
    '''
    def __init__(self, predictions: Array, real: Array, uncertainties=None):
        '''
        Initializes an instance of the metric class, including the
        predictions, uncertainties (optional), and real data
        necessary for comparison.

        Arg:
            predictions(ndarray):
                Array of predictions generated from the uncertainty model

            real(ndarray):
                Array of real values that you want to compare the uncertainty
                model ouput to (eg. experimental data)

            uncertainties(ndarray):
                Optional argument which contains array of uncertainty values
                generated from the uncertainty module
        '''
        self.predictions = predictions
        self.real = real
        self.uncertainties = uncertainties
        self.calculate()

    def calculate(self, **kwargs):
        '''
        Empty calculate function
        '''
        return


# Children classes for each relevent metric
class MSE(Metric):
    '''
    Mean Squared Error class which defines the structure used for
    computing the MSE between the passed in datasets. Inherets the
    properties of the parent class Metrics.
    '''

    def calculate(self, **kwargs) -> Tuple[float, Array]:
        '''
        Method that defines the mathematical formula necessary to compute the
        MSE.

            Args:

                **kwargs:
                    Necessary keyword arguments to be passed into calculate()
                    method

            Returns:

                MSE_value(float):
                    Total value of the MSE computed

                MSE_values(ndarray):
                    An array of MSE scores for each prediction

        '''
        MSE_value = np.square(np.subtract(self.real, self.predictions)).mean()
        MSE_values = []
        
        for i in range(len(self.predictions)):
            MSE_values.append((self.real[i] - self.predictions[i])**2)

        return MSE_value, MSE_values


class RMSE(Metric):
    '''
    Root Mean Squared Error class which defines the structure used for
    computing the RMSE between the passed in datasets. Inherets the
    properties of the parent class Metrics.
    '''

    def calculate(self, **kwargs) -> Tuple[float, Array]:
        '''
        Method that defines the mathematical formula necessary to compute the
        RMSE.

            Args:

                **kwargs:
                    Necessary keyword arguments to be passed into calculate()
                    method

            Returns:

                RMSE_value(float):
                    Total value of the RMSE computed

                RMSE_values(ndarray):
                    Array of RMSE values for each prediction

         '''
    # pseudocode
    # define mathematical formula for RMSE calculations using self.args
    # iteration over arrays and plug into defined formula
        RMSE_value = None
        RMSE_values = None
        return RMSE_value, RMSE_values


class F1(Metric):
    '''
    F1 score class which defines the structure used forcomputing the F1 score
    between the passed in datasets. Inherets the properties of the parent
    class Metrics.
    '''

    def calculate(self, **kwargs) -> float:
        '''
        Method that defines the mathematical formula necessary to compute
        the RMSE.

            Args:

                **kwargs:
                    Necessary keyword arguments to be passed into calculate()
                    method

                Returns:

                    F1_value(float):
                        Value of the F1 score computed

         '''
    # pseudocode
    # define mathematical formula for F1 calculation using self.args variables
    # iteration over arrays and plug into formula
        F1_value = None
        return F1_value
