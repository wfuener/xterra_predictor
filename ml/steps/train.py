"""
This module defines the following routines used by the 'train' step of the regression recipe:

- ``estimator_fn``: Defines the customizable estimator type and parameters that are used
  during training to produce a model recipe.
"""
from typing import Dict, Any
from sklearn.linear_model import LinearRegression


def train_model(train_data):
    regression = LinearRegression().fit(train_data[['bike_time']], train_data[['place']])
    # To retrieve the intercept:
    print(regression.intercept_)
    # For retrieving the slope:
    print(regression.coef_)
    return regression
