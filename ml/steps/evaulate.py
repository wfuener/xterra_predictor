import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error


def test_model(test_data, regression):
    predicted_place = regression.predict(test_data[['bike_time']])
    print(predicted_place.flatten())
    print(test_data['place'])
    df = pd.DataFrame({'Predicted': predicted_place.flatten(), 'Actual': test_data['place']})
    df.to_csv("artifacts/predicted_values.csv")
    mlflow.log_artifact("artifacts/predicted_values.csv")

    mse = mean_squared_error(y_true=test_data['place'], y_pred=predicted_place.flatten())
    mlflow.log_metric("mse", mse)
    print(f"mse: {mse}")

