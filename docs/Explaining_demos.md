# Demos can be found at GANdy/examples/

## Package_demo
	- Demonstrates all features of the gandy package
	- Trains all three models (GP, BNN, GAN) on synthetic noisy data
	- Evaluates estimated uncertaintities using the UCP metric
	- Compares predictions from three models to a Support Vector Machine regressor
	- Utilizes hyperparameter optimization on BNN

## BNN_demo
	- Demoonstrates a Bayes Neural Network (BNN) using the Boston dataset
	- Plots the predictions (with estimated uncertainties from the model) versus true target values

## GPs_demo
	- Demonstrates a Gaussian Process (GP) using the Boston dataset
	- Plots the predictions (with estimated uncertainties from the model) versus true target values
	- Flags predictions with uncertainties above a threshold

## gan_demo
	- Demonstrates a Generative Adversarial Network (GAN) using the Boston dataset
	- Plots the predictions (with estimated uncertainties from the model) versus true target values
	- Flags predictions with uncertainties above a threshold

## Metrics_demo 
	- Demonstrates two main regression metrics: 
		1. Mean Squared Error (MSE)
		2. Root Mean Squared Error (RMSE)
	- Demonstrates uncertainty metric
		1. Uncertainty Coverage Probability (UCP)

## qm9_noise_data.csv
	- QM9 data with random noise to test on models