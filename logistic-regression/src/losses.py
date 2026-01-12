import numpy as np


def log_loss(y, p, eps=1e-15):
    """
    Binary cross-entropy (log loss).

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    p : array-like of shape (n_samples,)
        Predicted probabilities for class 1
    eps : float
        Small value to avoid log(0)

    Returns
    -------
    loss : array-like
        Log loss per sample
    """
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def log_loss_gradient(y, p):
    """
    Gradient of log loss with respect to the model score z,
    assuming p = sigmoid(z).

    Core result:
        dL/dz = p - y

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True binary labels
    p : array-like of shape (n_samples,)
        Predicted probabilities

    Returns
    -------
    grad : array-like
        Gradient with respect to z
    """
    return p - y
