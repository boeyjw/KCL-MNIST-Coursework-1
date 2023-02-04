import numpy as np
import pandas as pd

def sgt_heaviside(x):
    """Heaviside function.

    Args:
        x (float): w @ Augmented X

    Returns:
        float: 0.5 if x == 0, 0 if x < 0 and 1 if x > 0
    """
    if x == 0:
        return 0.5
    elif x < 0:
        return 0
    elif x > 0:
        return 1

def sgt_seq_delta_learning(X, y, w, bias, eta=0.01, max_iter=100, as_frame=True):
    """Sequential Delta Learning Rule

    Args:
        X (np.array): X variables
        y (np.array): label
        w (np.array): weights
        bias (float): theta
        eta (float, optional): Learning rate. Defaults to 0.01.
        max_iter (int, optional): Maximum iterations to force terminate. Defaults to 100.
        as_frame (bool, optional): Return Pandas DataFrame. Defaults to True.

    Returns:
        (pd.DataFrame|list): yhat, weights, error
    """
    wb = np.concatenate(([-bias], w))
    i = 0
    converged = 0

    res = []
    while i < max_iter and converged != len(y):
        for j in range(len(y)):
            augX = np.insert(X[j, :], 0, 1)
            yhat = sgt_heaviside(wb @ augX)
            wb = wb + eta * (y[j] - yhat) * augX.T
            if (y[j] - yhat) == 0:
                converged = converged + 1
            else:
                converged = 0
            res.append((yhat, wb, y[j] - yhat))
            if converged == len(y):
                break
            i = i + 1
    return pd.DataFrame(res, columns=["yhat", "weights", "error"]) if as_frame else res

def sgt_seq_linear_threshold(X, y, w, bias):
    """Sequential Linear Threshold

    Args:
        X (np.array): X Variables
        y (np.array): labels
        w (np.array): Weights
        bias (float): theta

    Returns:
        pd.DataFrame: Confusion matrix
    """
    wb = np.concatenate(([-bias], w))
    yhat = np.array([wb.T @ np.insert(X[i, :], 0, 1) for i in range(len(y))])
    yhat[yhat > 0] = 1
    yhat[yhat <= 0] = 0
    return pd.DataFrame([
        [np.sum((yhat == 1) & (y == 1)), np.sum((yhat == 0) & (y == 1))],
        [np.sum((yhat == 1) & (y == 0)), np.sum((yhat == 0) & (y == 0))],
    ], columns=["P", "N"], index=["P", "N"])