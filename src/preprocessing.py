import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Apply one-hot encoding for categorical features
def encode_categorical_features(df, categorical_features):
  df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=False)
  return df_encoded


# Apply StandardScaler to numerical featuers
def feature_scaling(df, numerical_features):
  scaler = StandardScaler()
  df[numerical_features] = scaler.fit_transform(df[numerical_features])
  return df


# Split the data into train set and test set
def train_test_split(df, target_column):
  X = df.drop(target_column, axis=1)
  y = df[target_column]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
