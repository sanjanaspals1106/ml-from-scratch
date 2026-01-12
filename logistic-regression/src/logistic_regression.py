import numpy as np


class LogisticRegression:
    """
    Logistic Regression trained using Gradient Descent.
    Binary classification only.
    """

    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.losses = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _log_loss(self, y, p, eps=1e-15):
        p = np.clip(p, eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.n_iters):
            # Linear score
            z = X @ self.w + self.b

            # Probability prediction
            p = self._sigmoid(z)

            # Compute loss (mean)
            loss = np.mean(self._log_loss(y, p))
            self.losses.append(loss)

            # Gradients (core result from Day 5)
            dz = p - y
            dw = (1 / n_samples) * (X.T @ dz)
            db = (1 / n_samples) * np.sum(dz)

            # Parameter update
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        z = X @ self.w + self.b
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
