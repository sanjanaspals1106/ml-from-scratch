import numpy as np


class LinearRegression:
    """
    Linear Regression from scratch.

    Supports:
    - Normal Equation (closed-form)
    - Batch Gradient Descent (iterative)
    """

    def __init__(self):
        self.weights = None
        self.loss_history = []

    # ---------- Core utilities ----------

    def _add_bias(self, X):
        m = X.shape[0]
        return np.c_[np.ones((m, 1)), X]

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model has not been fitted yet.")
        X_b = self._add_bias(X)
        return X_b @ self.weights

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # ---------- Normal Equation ----------

    def fit_normal_equation(self, X, y):
        """
        Closed-form solution:
        w = (X^T X)^(-1) X^T y
        """
        X_b = self._add_bias(X)
        self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    # ---------- Gradient Descent ----------

    def fit_gradient_descent(self, X, y, lr=0.01, epochs=1000):
        """
        Batch Gradient Descent optimization.
        """
        X_b = self._add_bias(X)
        m, n = X_b.shape

        self.weights = np.zeros(n)
        self.loss_history = []

        for _ in range(epochs):
            y_pred = X_b @ self.weights
            errors = y_pred - y

            gradients = (2 / m) * X_b.T @ errors
            self.weights -= lr * gradients

            loss = self.mse(y, y_pred)
            self.loss_history.append(loss)
