from deae_utils import convert_matrix_to_vector, convert_vector_to_matrix
from sklearn.linear_model import LogisticRegression

def logit(x_train, y_train, x_test):
    if len(y_train.shape) > 1:
        y_train = convert_matrix_to_vector(y_train)

        # Define and fit model on training dataset
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Predict on x_test
    y_test_hat = model.predict_proba(x_test)

    return y_test_hat