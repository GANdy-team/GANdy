## Use cases
1. Creating uncertainty prediction models
	- *User* - Provide an architecture option, and hyperparameters specified
	- *Function* -  Initiate a model of the desired class
	- *Results* - Return the desired build instance
2. Checking data compatible with model
	- *User* - Provide data with or w/o target/s, and a model instance
	- *Function* - Assert model can be used to make predictions on or train on data
	- *Results* - No errors 
3. Make predictions: @ deepchem, keras, sklearn
	- *User* - Provide data, call upon a model instance
	- *Function* - Check data comparable with model, call the predict method in desired model on provided data
	- *Results* - Predictions of target and uncertainty estimation for given examples

4. Train a model: @ deepchem, keras, sklearn
	- *User* - Provide data, provide a target, call upon a model instance
	- *Function* - Check data compatible with model, call the fit method in desired model on provided data
	- *Results* - A model instance with trained parameters

5. Saving and loading models: @pickle and h5
	- *User* - Provide a file descriptor
	- *Function* - Save/store existing model instances for future use
	- *Results* - Pickled object

6. Optimize model architecture:
	- *User* - Provide the base model, data to fit to, desired targets, optimization scheme option
	- *Function* - Check data compatible with model, run optimization scheme for chosen base model on provided data
	- *Results* - Model with hyperparameters optimized to minimize loss **and a distribution/uncertainty metric**

7. Identification of costly predictions:
	- *User* - Provide data to make predictions on, and a value of uncertainty that is considered too costly
	- *Function* - Check data compatible with model, make predictions and estimations of uncertainty on data, filter out costly predictions
	- *Results* - Predictions with uncertain/costly examples flagged

8. Evaluation of uncertainty predictions
	- *User* - Provide dataset with predictions and uncertainties made
	- *Function* - Compare the uncertainties predicted to the dataset and expected variance, quantify the quality of the uncertainty predictions
	- *Results* - Value/Quality of predictions

9. Judgement of uncertainty:
	- *User* - Provide a trained model
	- *Function* - Analysis of feature space associated with high uncertainty
	- *Results* - Classes/regions of regression that the model is uncertain on, indicating the need for more data or better features

10. Model architecture comparison:
	- *User* - Provide data to tune to, a target, desired model architectures
	- *Function* - Train all architectures as specified, measure cost, and make predictions
	- *Results* - Predictions on data for each chosen architecture, with training and predicting cost
    
11. Create a model to predict a molecular feature and estimate the costly predictions
	- *User* - Provide training and testing data set, target of interest, a model architecture with desired hyperparameters, and a proclaimed value of uncertainty that is unacceptable
	- *Function* - Initiate the model with desired hyperparameters and shapes, train the model on provided training data, evaluate both loss and distribution metrics, make predictions and uncertainty estimation on testing data, and flag costly predictions
	- *Results* - A trained model, predictions of both training and testing data with examples of high uncertainty identified