import pandas as pd
from linear_regression import LinearRegression
from preprocessing import encode_categorical_features, feature_scaling, train_test_split


def load_data():
  df = pd.read_csv('./data/insurance.csv')
  df = encode_categorical_features(df, ['sex', 'smoker', 'region'])
  df = feature_scaling(df, ['age', 'bmi', 'children'])
  X_train, X_test, y_train, y_test = train_test_split(df, 'charges')


def train_linear_regression(X_train, y_train):

  model = LinearRegression()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)


if __name__ == "__main__":
  load_data()
  train_linear_regression()