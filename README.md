# GANdy [![Build Status](https://travis-ci.org/GANdy-team/GANdy.svg?branch=main)](https://travis-ci.org/GANdy-team/GANdy)
Automatically creating and comparing supervised machine learning models capable of returning uncertainty estimates in addition to predictions.

__Current Functionality__:
- [ ] Instantialize, train, and use uncertainty models
- - [ ] Gaussian Processes
- - [ ] Bayesian Neural Networks
- - [ ] uncertainty GANs
- [ ] Judge the quality of produced uncertainties with uncertainty metrics
- [ ] Model optimization to uncertainty metrics
- [ ] Comparison of model structures

## Installation
In console, execute the following command where <code>package_path</code> is the path to the folder containing this Readme (GANdy):
> <code>pip install package_path</code>
> 
It can then be imported on the installed environment as <code>gandy</code>.

## Repo structure
```
GANdy
-----
setup.py                  # package installation
environment.yml           # development environment
examples/                 
|-GPs_Showcase.ipynb      # demo of gaussian processes as an uncertainty model
gandy/
|-tests/
|-models/
| |-models.py             # package parent model class
| |-bnns.py               # bayesian neural nets as an uncertainty model
| |-dcgan.py              # helper functions for GANs
| |-gans.py               # GANs as an uncertainty model
| |-gps.py                # gaussian processes as an uncertainty model
|-quality_est/
| |-metrics.py            # tools for evaluating returned uncertainties and predictions

```

## Justification
For a supervised machine learning task, one generally obtains deterministic predictions of a target variable based on a learned relationship between that target and a set of features. Such models make predictions on new quantities idependant of known variability or lack of knowledge, and there is no idication of the quality of a prediction. For many tasks, where the target variable is sensative to small changes, it is important not only to have a prediction but also the uncertainty associated with the prediction, in order to inform prediction costs. 

Some models already exist that can approximate the uncertainty of a prediction, such as Gaussian Processes or Bayesian models, which have their own downsides including training cost. Recently (2020), it has been shown by Lee and Seok \[1\] that the relatively new architecture Generative Adversarial Networks (GAN) can formatted to produce distributions of a target conditions on features. Here, they invert the input and output of a traditional conditional GAN (cGAN) in order to make target predictions with uncertainty. 

It is desirable to have a tool in python that allows for the formulation of such uncertainty GANs, but also a comparison with other tools capable of predicting uncertainty. GANdy aims to incorporate these different tools and allow for automated optimization and comparison such that a model ideal for a task's cost to quality ratio can be identified. 

\[1\] M. Lee and J. Seok, “Estimation with Uncertainty via Conditional Generative Adversarial Networks.” ArXiv 2007.00334v1

## Examples
See <examples> for demonstrations on predicting uncertainties with the available tools.

## For developers
### Installation
To install the development environment <code>conda env create --file environment.yml</code>.
If any new installed development dependancies, add them to the environment.yml environment file by Manually adding the dependency, eg. 
>  \- python=3.6.*

To update dev environment with new dependencies in the .yml file, <code>conda env update --file environment.yml</code>

./working/ is a workspace for notebooks/testing. It will be ignored by git by default, and will be removed upon release. To specifically "save" your files to git or to share work with other developers, use <code>git add --force working</code>.

### Testing
Tests located at <gandy/tests>
