# AdaBoost - Custom Implementation

This repository contains a Python implementation of the AdaBoost algorithm created from scratch. The AdaBoost (Adaptive Boosting) algorithm is a powerful ensemble technique used to improve the performance of machine learning models by combining multiple weak learners to create a strong learner. This project utilizes simple classifiers like Decision Trees and Naive Bayes as weak learners.

## Project Description

The AdaBoost algorithm is designed to increase the accuracy of any given machine learning algorithm. It focuses on training models sequentially, each trying to correct its predecessor's mistakes. The final model is a weighted sum of these weak learners, capable of achieving high accuracy on both training and unseen data.

In this project, we implement AdaBoost using Python to understand the core mechanics such as weight updating, error calculation, and learner aggregation, all from scratch.

## Features

- **Custom AdaBoost Implementation**: Functions to manage the entire boosting process including training weak learners, updating instance weights, and aggregating predictions.
- **Versatility in Model Choice**: Flexibility to use either Decision Trees or Naive Bayes classifiers as the base learners.
- **Performance Evaluation**: Code to assess and visualize the accuracy of individual models as well as the entire ensemble.
- **Parameter Tuning**: Additional scripts for parameter optimization and model validation.

## Getting Started

### Prerequisites

Ensure you have Python installed on your machine. The project also requires the following Python libraries:
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

You can install these packages using pip:
`pip install pandas numpy matplotlib scikit-learn`


### Installation

Clone this repository to your local machine to get started:
`git clone https://github.com/your-username/AdaBoost-Project.git`
