# GANdy
This repository estimates uncertainty using GANs and other machine learning models such as  GPs and BNNs.

## Installation
In console, execute the following command where <code>package_path</code> is the path to the folder containing this Readme (GANdy):
> <code>pip install package_path</code>
It can then be imported on the installed environment as <code>gandy</code>.

## For developers
To install the development environment <code>conda env create --file devenv.yml</code>.
If any new installed development dependancies, add them to the devenv.yml environment file by Manually adding the dependency, eg. 
>  - python=3.6.*
To update dev environment with new dependencies in the .yml file, <code>conda env update --file devenv.yml</code>

./working/ is a workspace for notebooks/testing. It will be ignored by git by default, and will be removed upon release. To specifically "save" your files to git or to share work with other developers, use <code>git add --force working</code>.
