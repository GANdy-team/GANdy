# GANdy [![Build Status](https://travis-ci.org/GANdy-team/GANdy.svg?branch=main)](https://travis-ci.org/GANdy-team/GANdy) [![Coverage Status](https://coveralls.io/repos/github/GANdy-team/GANdy/badge.svg?branch=main)](https://coveralls.io/github/GANdy-team/GANdy?branch=main)
Automatically creating and comparing supervised machine learning models capable of returning uncertainty estimates in addition to predictions.

__Current Functionality__:
- [x] Instantialize, train, and use uncertainty models
- - [x] Gaussian Processes
- - [x] Bayesian Neural Networks
- - [x] uncertainty GANs
- [x] Judge the quality of produced uncertainties with uncertainty metrics
- [ ] Automated comparison of model structures
- [ ] Model optimization to uncertainty metrics

## Installation
Install and activate the environment with `environment.yml` by:
> `conda env create -f environment.yml`

> `conda activate gandy_env`

In console, execute the following command where <code>package_path</code> is the path to the folder containing this Readme (GANdy):
> <code>pip install package_path</code>

It can then be imported on the installed environment as <code>gandy</code>.

## Repo structure
```
GANdy
-----
setup.py                  # package installation
environment.yml           # environment
devenv.yml                # development environment - contains packages for plotting
examples/                 
|-BNN_demo.ipynb          # demo of bayensian NN as an uncertainty model
|-GPs_Showcase.ipynb      # demo of gaussian processes as an uncertainty model
|-Metrics_demo.ipynb      # demonstration of using gandy metrics
|-Package_demo.ipynb      # showcase of current package functionality
|-GAN_demo.ipynb          # demo of GANs as an uncertainty model
gandy/
|-tests/
|-models/
| |-models.py             # package parent model class
| |-bnns.py               # bayesian neural nets as an uncertainty model
| |-dcgan.py              # helper functions for GANs
| |-gans.py               # GANs as an uncertainty model
| |-gps.py                # gaussian processes as an uncertainty model
|-quality_est/
| |-datagen.py            # functions to generate synthetic uncertainty data
| |-metrics.py            # tools for evaluating returned uncertainties and predictions
|-optimization/
| |-hypersearch.py        # tools for hyperparameter optimization

```

## Justification
For a supervised machine learning task, one generally obtains deterministic predictions of a target variable based on a learned relationship between that target and a set of features. Such models make predictions on new quantities idependant of known variability or lack of knowledge, and there is no idication of the quality of a prediction. For many tasks, where the target variable is sensative to small changes, it is important not only to have a prediction but also the uncertainty associated with the prediction, in order to inform prediction costs. 

Some models already exist that can approximate the uncertainty of a prediction, such as Gaussian Processes or Bayesian models, which have their own downsides including training cost. Recently (2020), it has been shown by Lee and Seok \[1\] that the relatively new architecture Generative Adversarial Networks (GAN) can formatted to produce distributions of a target conditions on features. Here, they invert the input and output of a traditional conditional GAN (cGAN) in order to make target predictions with uncertainty. 

It is desirable to have a tool in python that allows for the formulation of such uncertainty GANs, but also a comparison with other tools capable of predicting uncertainty. GANdy aims to incorporate these different tools and allow for automated optimization and comparison such that a model ideal for a task's cost to quality ratio can be identified. 

\[1\] M. Lee and J. Seok, “Estimation with Uncertainty via Conditional Generative Adversarial Networks.” ArXiv 2007.00334v1

## Examples
![image](https://drive.google.com/uc?export=view&id=1Sm-3Imu2sNLvcRof2qscVS8dU0L5W0uC)

See <examples> for more demonstrations on predicting uncertainties with the available tools.

## For developers
### Installation
To install the development environment <code>conda env create --file devenv.yml</code>.
If any new installed development dependancies, add them to the environment.yml environment file by Manually adding the dependency, eg. 
>  \- python=3.6.*

To update dev environment with new dependencies in the .yml file, <code>conda env update --file environment.yml</code>

./working/ is a workspace for notebooks/testing. It will be ignored by git by default, and will be removed upon release. To specifically "save" your files to git or to share work with other developers, use <code>git add --force working</code>.

### Testing
Tests located at <gandy/tests>
