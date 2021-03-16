## Component spec

### <span style="color:green">Implimented:</span>

1. `gandy.models.models.UncertaintyModel`
    - *Use cases* - (1, 2, 7, 8)
    - *Description* - Parent class to provide foundation for all uncertainty models.
    - *Key methods* - `__init__`, `check_data`, `train`, `predict`, `score`
    - *Key attributes* - `metrics`: metrics in `gandy.quality_est.metrics`
    - *Inputs* - Data shape, hyperparameters, X/Y data, acceptable uncertainty
    - *Outputs* - self (not implimented model)
    
2. `gandy.models.gps.ucGaussianProcess` (UncertaintyModel) @sklearn
    - *Use cases* - (3, 4)
    - *Description* - Child class defining gaussian process uncertainty models
    - *Key methods* - `_build`, `_train`, `_predict`, `R`, `C`
    - *Inputs* - Data shape, hyperparameters, X/Y data, acceptable uncertainty, learning type (classification or regression)
    - *Outputs* - self (trained gaussian process model), predictions, uncertainties, score of model on test data
    
3. `gandy.models.bnns.BNN` (UncertaintyModel) @keras
    - *Use cases* - (3, 4)
    - *Description* - Child class defining bayesian NN uncertainty models
    - *Key methods* - `_build`, `_train`, `_predict`, `prior`, `posterior`, `negative_logliklihood`
    - *Inputs* - Data shape, hyperparameters, X/Y data, acceptable uncertainty, training set size
    - *Outputs* - self (trained bayesian NN), predictions, uncertainties, score of model on test data
    
4. `gandy.models.gans.GAN` (UncertaintyModel) @deepchem
    - *Use cases* - (3, 4)
    - *Description* - Child class defining GAN uncertainty models
    - *Key methods* - `_build`, `_train`, `_predict`, `iter_batches`
    - *Inputs* - Data shape, hyperparameters, X/Y data, acceptable uncertainty
    - *Outputs* - self (trained GAN), predictions, uncertainties, score of model on test data
    
5. `gandy.quality_est.metrics.Metric` (and subclasses)
    - *Use cases* - (4, 6, 10)
    - *Description* - Class defining available metrics, eg. MSE, UCP
    - *Key methods* - `calculate`
    - *Inputs* - true values, prediction, uncertainties
    - *Outputs* - evaluated overall metric and example metric vector

6. `gandy.quality_est.datagen` @deepchem
    - *Use cases* - (6, 8, 10)
    - *Description* - Functions to generate analytical synthetic data with known noise, and data of QM9 experimental data with added known noise
    - *Inputs* - whether to save to csv, features in qm9 to use
    - *Outputs* - generated datasets
    
7. `gandy.optimization.hypersearch.OptRoutine` @optuna
    - *Use cases* - (6)
    - *Description* - Hyperparameter optimization routine
    - *Key methods* - `optimize`, `train_best`
    - *Key attributes* - `search_space`: the search space for parameters
    - *Inputs* - model to optimize, foundational hyperparameters, hyperparameters to search, development data, validation scheme, optimization scheme, pruning parameters
    - *Outputs* - search results, best hyperparameters, best model
    
### <span style="color:green">Not Implimented:</span>

8. `gandy.quality_est.ModelComparison`
    - *Use cases* - (9, 10)
    - *Description* - Evaluation of models against the data and eachother
    - *Inputs* - already trained models or an OptRoutine to create a trained model, data, metrics to use
    - *Outputs* - quantitiative comparison of models, data space with poor certainty