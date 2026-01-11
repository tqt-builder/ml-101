import pickle
import numpy as np
import pandas as pd
from linear_regression import LinearRegression
from preprocessing import encode_categorical_features, feature_scaling, split_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import argparse
from neural_network import SimpleNeuralNetwork
import os


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

def train_neural_network(X_train, y_train, X_test, y_test):
    n_features = X_train.shape[1]
    
    # 1. Scale Target Variable (y)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    
    # Save the target scaler (needed for API to return actual dollar amounts)
    with open('models/scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)

    # Initialize NN: 2 layers, 5 neurons each
    model = SimpleNeuralNetwork(n_features=n_features, learning_rate=0.001, n_iters=20000)
    
    print("\nStarting Neural Network Training (Target Scaled)...")
    model.fit(X_train.values, y_train_scaled)
    
    # 3. Predict and Inverse Transform (unscale) back to dollars
    y_pred_scaled = model.predict(X_test.values)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nNeural Network Model Performance:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")
    
    model.save('models/neural_network')


if __name__ == "__main__":
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Train insurance prediction models.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="linear", 
        choices=["linear", "nn"],
        help="Choose model to train: 'linear' or 'nn' (default: linear)"
    )
    args = parser.parse_args()

    # 2. Load Data
    X_train, X_test, y_train, y_test = load_data()
    y_test_reshaped = y_test.values.reshape(-1, 1)

    # 3. Execute chosen training
    if args.model == "linear":
        train_linear_regression(X_train, y_train, X_test, y_test_reshaped)
    elif args.model == "nn":
        train_neural_network(X_train, y_train, X_test, y_test_reshaped)