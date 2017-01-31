# GaussianDecisionBoundaries
Python script to illustrate decision boundaries between two bivariate Gaussian distributions

I wrote this script as an illustration of a Maximum a Posteriori classifier based on Gaussian distributions, mainly for teaching purposes. You can change a priori probabilities, mean vectors, covariance matrices. You can choose to display the (hard) decision boundary or the (soft) difference in posterior probabilities. You can also display contours for each distribution. Every time you change parameters, click on Redraw to update the plot.

TODO:
* update axis range based on Gaussian parameters
* gracefully complain if the covariance matrix is not definite positive
* clean up/simplify the code
