import pandas as pd
from sklearn import model_selection, linear_model, metrics

def test_model_initialization():
    model = linear_model.LinearRegression()
    assert isinstance(model, linear_model.LinearRegression), \
        "Failed to create LinearRegression instance"

def test_prediction_shape():
    test_input = [[0], [1], [2]]  
    model = linear_model.LinearRegression().fit(test_input, [10, 20, 30])
    predictions = model.predict(test_input)
    assert len(predictions) == 3, \
        f"Expected 3 predictions, got {len(predictions)}"

def test_mse_non_negative():
    y_true = [100, 200, 300]
    y_pred = [110, 190, 310]
    mse = metrics.mean_squared_error(y_true, y_pred)
    assert mse >= 0, f"Impossible negative MSE: {mse}"

test_model_initialization()
test_prediction_shape()
test_mse_non_negative()