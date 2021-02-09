# Logistic-Regression-With-Numpy

Implementation of Logistic Regression from Scratch Using Numpy. Implemented as a homework for CMPE544 Pattern Recognition Course of Boğaziçi University, tested on a toy dataset provided by the university (unfortunately it cannot be added to this repository).

In this implementation, stochastic gradient descent was used by calculating new weights for a random data point at each iteration, going over all data points at one epoch. 
In order to do that, data points and labels are shuffled at the beginning of each epoch and went over them in an in order fashion. 
This approach made sure that a random data point is choosen to calculate weights on each iteration, the same datapoint is not choosen twice in a single epoch and we go over all datapoints on one epoch. Logistic loss is used as the loss function and recorded at the end of each epoch.
