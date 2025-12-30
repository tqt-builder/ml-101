import pickle
import numpy as np
import pandas as pd
from linear_regression import LinearRegression
from preprocessing import encode_categorical_features, feature_scaling, split_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data():
  df = pd.read_csv('./data/insurance.csv')
  df = encode_categorical_features(df, ['sex', 'smoker', 'region'])
  df, scaler = feature_scaling(df, ['age', 'bmi', 'children'])
  with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
  with open('models/columns.pkl', 'wb') as f:
    pickle.dump(df.drop('charges', axis=1).columns.tolist(), f)
  return split_data(df, 'charges')


def train_linear_regression(X_train, y_train, X_test, y_test):
  model = LinearRegression()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  # loss = np.mean((y_pred - y_test)**2)
  mae_linear = mean_absolute_error(y_test, y_pred)
  mse_linear = mean_squared_error(y_test, y_pred)
  r2_linear = r2_score(y_test, y_pred)

  # Print the calculated metrics
  print("Linear Regression Model Performance:")
  print(f"Mean Absolute Error (MAE): {mae_linear:.2f}")
  print(f"Mean Squared Error (MSE): {mse_linear:.2f}")
  print(f"R-squared (R2): {r2_linear:.2f}")

  model.save()


if __name__ == "__main__":
  X_train, X_test, y_train, y_test = load_data()
  y_test = y_test.values.reshape(-1, 1)
  train_linear_regression(X_train, y_train, X_test, y_test)