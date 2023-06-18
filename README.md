# Ada-Boost custom implementation 

This repository contains a custom implementation of the AdaBoost (Adaptive Boosting) algorithm. The purpose of this project is to provide an in-depth understanding of the underlying logic behind the AdaBoost algorithm, without the use of existing Python libraries.

## About AdaBoost

AdaBoost is a powerful ensemble machine learning algorithm, mainly used to improve the results of decision trees on binary classification problems. It works by creating a strong classifier from a number of weak classifiers. This is done by building a model from the training data, then creating a second model that attempts to correct the errors from the first model. Models are added until the training set is predicted perfectly or a maximum number of models are added.

### Repository Structure

`adaboost_jovanailin.py`: This is the main Python script containing the implementation of the AdaBoost algorithm.

`drugY.csv`: This CSV file contains the dataset used for testing the implementation.

#### Prerequisites
Python 3.x
