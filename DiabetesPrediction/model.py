# model.py

import numpy as np

class NeuralNetwork:
    def __init__(self, input_layer, hidden_layer, output_layer):
        self.W1 = np.random.rand(input_layer, hidden_layer)
        self.b1 = np.random.rand(1, hidden_layer) * 0.01
        self.W2 = np.random.rand(hidden_layer, output_layer)
        self.b2 = np.random.rand(1, output_layer) * 0.01

    def reLu(self, z):
        return np.maximum(0, z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward_prop(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.reLu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def compute_loss(self, y, y_hat):
        loss = -np.mean(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))
        return loss

    def back_prop(self, X, y, learning_rate):
        m = X.shape[0]
        dZ2 = self.A2 - y
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            y_hat = self.forward_prop(X)
            loss = self.compute_loss(y, y_hat)
            self.back_prop(X, y, learning_rate)

    def predict(self, X):
        prob = self.forward_prop(X)
        pred = np.round(prob)
        return pred, prob
