# GaussianDecisionBoundaries

![Illustration](https://github.com/giampierosalvi/GaussianDecisionBoundaries/blob/master/GaussianDecisionBoundaries.png "")

Python script to illustrate decision boundaries between two bivariate Gaussian distributions

To run first install dependencies:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
then run:
```
python3 GaussianDecisionBoundaries.py
```

I wrote this script as an illustration of a Maximum a Posteriori classifier based on Gaussian distributions, mainly for teaching purposes. You can change a priori probabilities, mean vectors, covariance matrices. You can choose to display the (hard) decision boundary or the posterior for one class, the scaled posteriors or the log-likelihood ratio. You can also display contours for each distribution. Every time you change parameters, press Enter or click on Redraw to update the plot. The program will gracefully complain if you input a covariance matrix that is not positive definite.

NOTE: if you use the Python interpreter from Anaconda, you might get very ugly looking fonts. This is due to a limitation of the compiled version of the package tkinter that comes with Anaconda. You may try to use other versions of Python instead to get better results.

TODO:
* update axis range based on Gaussian parameters
* clean up/simplify the code

## Jupyter Notebook version
The file `GaussianDecisionBoundaries_notebook.py` will contain a Jupyter notebook implementation if that is more convenient. This file has been obtained with `jupytext` from an `.ipynb` file and can be opened directly in Jupyter.
