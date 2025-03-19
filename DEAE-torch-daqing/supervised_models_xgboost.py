import numpy as np

import xgboost as xgb
from deae_utils import convert_matrix_to_vector, convert_vector_to_matrix


def xgb_model(x_train, y_train, x_test):
    """XGBoost.

    Args:
      - x_train, y_train: training dataset
      - x_test: testing feature

    Returns:
      - y_test_hat: predicted values for x_test
    """
    # Convert labels into proper format
    if len(y_train.shape) > 1:
        y_train = convert_matrix_to_vector(y_train)

        # Define and fit model on training dataset
    model = xgb.XGBClassifier()
    model.fit(x_train, y_train)

    # Predict on x_test
    y_test_hat = model.predict_proba(x_test)

    return y_test_hat