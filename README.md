# GANdy
This repository estimates uncertainty using GANs and other machine learning models such as  GPs and BNNs.


### Use cases
1. Creating uncertainty prediction models
	- *User* - Provide an architecture option, and hyperparameters specified
	- *Function* -  initiate a model of the desired class
	- *Results* - return the desired build instance
2. Checking data compatible with model
	- *User* - Provide data with target/s, and a model instance
	- *Function* - assert model can be used to make predictions on or train on data
	- *Results* - No errors 
3. Make predictions: @ deepchem
	- *User* - Provide data, provide a target, call upon a model instance
	- *Function* - check data comparable with model, call the predict method in desired model on provided data
	- *Results* - predictions of target and uncertainty estimation for given examples

4. Train a model: @ deepchem
	- *User* - Provide data, provide a target, call upon a model instance
	- *Function* - call the fit method in desired model on provided data
	- *Results* - a model instance with trained parameters

5. Saving and loading models: @pickle and h5
	- *User* - Provide a format, and file descriptor
	- *Function* - save/store existing model instances for future use
	- *Results* - pickled object

6. Optimize model architecture:
	- *User* - Provide the base model, data to fit to, desired targets, optimization scheme option
	- *Function* - Run optimization scheme for chosen base model on provided data
	- *Results* - Model with hyperparameters optimized to minimize loss **and a distribution metric (How will we do this?? Needs exploration)**

7. Identification of costly predictions:
	- *User* - Provide data to make predictions on, and a value of uncertainty that is considered too costly
	- *Function* - Make predictions and estimations of uncertainty on data, filter out costly predictions
	- *Results* - Predictions with uncertain/costly examples flagged

8. Judgement of uncertainty:
	- *User* - provide data to make predictions on, and a model
	- *Function* - light analysis of feature space associated with high uncertainty
	- *Results* - Classes/regions of regression that the model is uncertain on, indicating the need for more data or better features

9. Model architecture comparison:
	- *User* - provide data to tune to, a target, desired model architectures
	- *Function* - train all architectures as specified, measure cost, and make predictions
	- *Results* - Predictions on data for each chosen architecture, with training and predicting cost
