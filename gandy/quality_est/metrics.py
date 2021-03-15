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
import warnings

from sklearn.metrics import f1_score, accuracy_score

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
        return

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
        if self.uncertainties is not None:
            warnings.warn("MSE metric does not take uncertainties as arg")


        # Define MSE formula using numpy methods
        MSE_value = np.mean(np.square(np.subtract(self.real, self.
                                                  predictions)))

        # Define MSE_values as a list of MSE deviations between each data
        # point
        MSE_values = []

        # Iterate through data points and add MSE value to list
        for i in range(len(self.predictions)):
            MSE_values.append((self.real[i] - self.predictions[i])**2)

        return (MSE_value, MSE_values)


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
        if self.uncertainties is not None:
            warnings.warn("RMSE metric does not take uncertainties as arg")

        # Define RMSE using numpy methods
        RMSE_value = np.sqrt(np.mean(np.subtract(self.real, self.
                                                 predictions)**2))

        # Define RMSE_values as a list of RMSE deviations between data
        # points
        RMSE_values = []

        for i in range(len(self.predictions)):
            RMSE_values.append(np.sqrt((self.real[i] - self.predictions[i]
                                        )**2))

        return (RMSE_value, RMSE_values)


class F1(Metric):
    '''
    F1 score class which defines the structure used for computing the F1 score
    between the passed in datasets. Inherets the properties of the parent
    class Metrics.
    '''

    def calculate(self, **kwargs) -> float:
        '''
        Method that defines the mathematical formula necessary to compute
        the F1 score.
            Args:
                **kwargs:
                    Necessary keyword arguments to be passed into calculate()
                    method
                Returns:
                    F1_value(float):
                        Value of the F1 score computed
         '''
        if self.uncertainties is not None:
            warnings.warn("F1 metric does not take uncertainties as arg")

        F1_value = f1_score(self.real, self.predictions, average='macro')
        return (F1_value, None)


class Accuracy(Metric):
    '''
    Accuracy class which defines the structure used for computing the
    accuracy score between the passed in datasets. Inherets the properties of
    the parent class Metrics.
    '''

    def calculate(self, **kwargs) -> float:
        '''
        Method that defines the mathematical formula necessary to compute
        the Accuracy.
            Args:
                **kwargs:
                    Necessary keyword arguments to be passed into calculate()
                    method
                Returns:
                    Accuracy_value(float):
                        Value of the Accuracy score computed
         '''
        if self.uncertainties is not None:
            raise TypeError("Accuracy metric does not take uncertainties as\
                arg")
        else:
            Accuracy_value = accuracy_score(self.real, self.predictions)
        return (Accuracy_value, None)


class UCP(Metric):
    '''
    Uncertainty coverage probability class which defines the structure used
    for computing the UCP between the passed in datasets. Inherets the
    properties of the parent class Metrics.
    '''

    def calculate(self, **kwargs) -> float:
        '''
        Method that defines the mathematical formula necessary to compute
        the UCP.
            Args:
                **kwargs:
                    Necessary keyword arguments to be passed into calculate()
                    method
                Returns:
                    UCP_value(float):
                        Value of the UCP score computed
         '''
        if self.uncertainties is None:
            warnings.warn("UCP metric requires uncertainties as arg")

        UCP_value = 0
        for i in range(len(self.predictions)):
            if ((self.predictions[i] - self.uncertainties[i]) <=
                    self.real[i] and
                (self.predictions[i] + self.uncertainties[i]) >=
                    self.real[i]):
                UCP_value += 1
            else:
                pass

        return (UCP_value*(1/len(self.predictions)), None)
