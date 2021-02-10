# GANdy
This repository estimates uncertainty using GANs and other machine learning models such as  GPs and BNNs.

## Installation

In console, execute the following command where <code>package_path</code> is the path to the folder containing this Readme (GANdy):
> <code>pip install package_path</code>
It can then be imported on the installed environment as <code>gandy</code>.

### Use cases
1. Creating uncertainty prediction models
	- *User* - Provide an architecture option, and hyperparameters specified
	- *Function* -  Initiate a model of the desired class
	- *Results* - Return the desired build instance
2. Checking data compatible with model
	- *User* - Provide data with target/s, and a model instance
	- *Function* - Assert model can be used to make predictions on or train on data
	- *Results* - No errors 
3. Make predictions: @ deepchem
	- *User* - Provide data, provide a target, call upon a model instance
	- *Function* - Check data comparable with model, call the predict method in desired model on provided data
	- *Results* - Predictions of target and uncertainty estimation for given examples

4. Train a model: @ deepchem
	- *User* - Provide data, provide a target, call upon a model instance
	- *Function* - Check data compatible with model,, call the fit method in desired model on provided data
	- *Results* - A model instance with trained parameters

5. Saving and loading models: @pickle and h5
	- *User* - Provide a format, and file descriptor
	- *Function* - Save/store existing model instances for future use
	- *Results* - Pickled object

6. Optimize model architecture:
	- *User* - Provide the base model, data to fit to, desired targets, optimization scheme option
	- *Function* - Check data compatible with model, run optimization scheme for chosen base model on provided data
	- *Results* - Model with hyperparameters optimized to minimize loss **and a distribution metric (How will we do this?? Needs exploration)**

7. Identification of costly predictions:
	- *User* - Provide data to make predictions on, and a value of uncertainty that is considered too costly
	- *Function* - Check data compatible with model, make predictions and estimations of uncertainty on data, filter out costly predictions
	- *Results* - Predictions with uncertain/costly examples flagged

8. Judgement of uncertainty:
	- *User* - Provide data to make predictions on, and a model
	- *Function* - Check data compatible with model, light analysis of feature space associated with high uncertainty
	- *Results* - Classes/regions of regression that the model is uncertain on, indicating the need for more data or better features

9. Model architecture comparison:
	- *User* - Provide data to tune to, a target, desired model architectures
	- *Function* - Train all architectures as specified, measure cost, and make predictions
	- *Results* - Predictions on data for each chosen architecture, with training and predicting cost
10. Create a model to predict a molecular feature and estimate the costly predictions
	- *User* - Provide training and testing data set, target of interest, a model architecture with desired hyperparameters, and a proclaimed value of uncertainty that is unacceptable
	- *Function* - Initiate the model with desired hyperparameters and shapes, train the model on provided training data, evaluate both loss and distribution metrics, make predictions and uncertainty estimation on testing data, and flag costly predictions
	- *Results* - A trained model, predictions of both training and testing data with examples of high uncertainty identified
