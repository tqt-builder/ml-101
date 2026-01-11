import numpy as np


class LinearRegression:
  def __init__(self, learning_rate=0.001, n_iters=10000):
    self.learning_rate = learning_rate
    self.n_iters = n_iters
    self.weight = None
    self.bias = None

  def fit(self, X, y):
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    n_samples, n_features = X.shape
    self.weight =  np.random.randn(n_features, 1)
    self.bias = 0

    y = y.reshape(-1, 1)
    for i in range(self.n_iters):
      y_p = np.dot(X, self.weight) + self.bias
      
      # Calculate Loss (Mean Squared Error)
      loss = np.mean((y_p - y)**2)
      
      if i % 100 == 0:
        print(f"Iteration {i}: Loss {loss:.4f}")

      dw = (1/n_samples) * np.dot(X.T, (y_p - y))
      db = (1/n_samples) * np.sum(y_p - y)

      self.weight -= self.learning_rate * dw
      self.bias -= self.learning_rate * db
  
  def predict(self, x):
    return np.dot(x, self.weight) + self.bias
  
  def save(self):
    np.save('models/linear_regression/y_array.pkl.npy', self.weight)
    np.save('models/linear_regression/bias.pkl.npy', self.bias)
  
  def load(self):
    self.weight = np.load('models/linear_regression/y_array.pkl.npy')
    self.bias = np.load('models/linear_regression/bias.pkl.npy')
