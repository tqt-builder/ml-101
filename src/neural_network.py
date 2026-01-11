import numpy as np
import os


class SimpleNeuralNetwork:
    def __init__(self, n_features, learning_rate=0.01, n_iters=20000):
        self.lr = learning_rate
        self.n_iters = n_iters
        
        # 1. Initialize Weights and Biases with He Initialization
        # Layer 1: Input -> 5 neurons
        self.W1 = np.random.randn(n_features, 5) * np.sqrt(2. / n_features)
        self.b1 = np.zeros((1, 5))
        
        # Layer 2: 5 neurons -> 5 neurons
        self.W2 = np.random.randn(5, 5) * np.sqrt(2. / 5)
        self.b2 = np.zeros((1, 5))
        
        # Output Layer: 5 neurons -> 1 neuron (Estimated Charge)
        self.W3 = np.random.randn(5, 1) * np.sqrt(2. / 5)
        self.b3 = np.zeros((1, 1))

    def relu(self, Z):
        """Rectified Linear Unit Activation"""
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        """Gradient of ReLU"""
        return (Z > 0).astype(float)

    def forward(self, X):
        """Forward Pass: Compute predictions"""
        # Layer 1
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        # Layer 2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)
        
        # Output Layer (Linear activation for regression)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        return self.Z3

    def backward(self, X, y, y_hat):
        """Backward Pass: Compute gradients using Chain Rule"""
        n_samples = X.shape[0]
        
        # 1. Output Layer Gradient
        dZ3 = y_hat - y # Shape: (n, 1)
        dW3 = (1 / n_samples) * np.dot(self.A2.T, dZ3)
        db3 = (1 / n_samples) * np.sum(dZ3, axis=0, keepdims=True)
        
        # 2. Layer 2 Gradient
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self.relu_derivative(self.Z2)
        dW2 = (1 / n_samples) * np.dot(self.A1.T, dZ2)
        db2 = (1 / n_samples) * np.sum(dZ2, axis=0, keepdims=True)
        
        # 3. Layer 1 Gradient
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = (1 / n_samples) * np.dot(X.T, dZ1)
        db1 = (1 / n_samples) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights and biases (Gradient Descent)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).reshape(-1, 1)
        
        for i in range(self.n_iters):
            y_hat = self.forward(X)
            self.backward(X, y, y_hat)
            
            if i % 1000 == 0:
                loss = np.mean((y_hat - y)**2)
                print(f"Iteration {i}: Loss {loss:.4f}")

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        return self.forward(X)

    def save(self, path='models/neural_network'):
        """Saves all weights and biases to the specified directory"""
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(f"{path}/W1.npy", self.W1)
        np.save(f"{path}/b1.npy", self.b1)
        np.save(f"{path}/W2.npy", self.W2)
        np.save(f"{path}/b2.npy", self.b2)
        np.save(f"{path}/W3.npy", self.W3)
        np.save(f"{path}/b3.npy", self.b3)
        print(f"Model saved to {path}")

    def load(self, path='models/neural_network'):
        """Loads weights and biases from the specified directory"""
        self.W1 = np.load(f"{path}/W1.npy")
        self.b1 = np.load(f"{path}/b1.npy")
        self.W2 = np.load(f"{path}/W2.npy")
        self.b2 = np.load(f"{path}/b2.npy")
        self.W3 = np.load(f"{path}/W3.npy")
        self.b3 = np.load(f"{path}/b3.npy")