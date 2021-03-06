# GaussianDecisionBoundaries

![Illustration](https://github.com/giampierosalvi/GaussianDecisionBoundaries/blob/master/GaussianDecisionBoundaries.png "")

Python script to illustrate decision boundaries between two bivariate Gaussian distributions

I wrote this script as an illustration of a Maximum a Posteriori classifier based on Gaussian distributions, mainly for teaching purposes. You can change a priori probabilities, mean vectors, covariance matrices. You can choose to display the (hard) decision boundary or the posterior for one class, the scaled posteriors or the log-likelihood ratio. You can also display contours for each distribution. Every time you change parameters, press Enter or click on Redraw to update the plot. The program will gracefully complain if you input a covariance matrix that is not positive definite.

NOTE: if you use the Python interpreter from Anaconda, you might get very ugly looking fonts. This is due to a limitation of the compiled version of the package tkinter that comes with Anaconda. You may try to use other versions of Python instead to get better results.

TODO:
* update axis range based on Gaussian parameters
* clean up/simplify the code
