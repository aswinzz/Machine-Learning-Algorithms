***Principal Component Analysis (PCA)*** is a statistical procedure that uses an orthogonal transformation which converts a set of correlated variables to a set of uncorrelated variables. PCA is a most widely used tool in exploratory data analysis and in machine learning for predictive models. Moreover, PCA is an unsupervised statistical technique used to examine the interrelations among a set of variables. It is also known as a general factor analysis where regression determines a line of best fit.

***Linear Discriminant Analysis (LDA)*** Linear discriminant analysis is a linear classifier used primarily for multiclass problems which aims to find a new feature space to project the data in order to maximize classes separability , it can be used to perform supervised dimensionality reduction.
Assumptions: Data should be gaussian distributed , each input variable should have same variance.
It can be easily implemented as a probabilistic model : By extracting mean and covariance from data and feeding it in Bayes theorem by plugging probability of x being in each class to a gaussian dist. function .
We get a discriminative function made of probability dist of classes,mean and covariance,and input and the class with highest value of discriminative function is chosen.
Extensions to LDA:
Quadratic DA:Each class uses its own estimate of covariance
Flexible Discriminant Analysis: non-linear combinations of inputs is used

***Difference between PCA and LDA***
PCA aims to find the attributes accounting for most of the variance in data and does not need class labels to work whereas LDA aims to identify attributes that account for the most variance between classes and is a supervised method.
