import pandas as pd
import numpy as np

def sgt_seq_widrow_hoff(X, y, w, bias, b, eta, max_iter=100, theta=0, as_frame=True, dec=4):
    """Sequential Widrow-Hoff Learning

    Args:
        X (np.array): X variables
        y (np,array): label
        w (np.array): weight
        bias (float): bias value
        b (np.array): margin vector
        eta (float): learning rate
        max_iter (int): maximum number of iterations
        theta (float): threshold for convergence
        as_frame (bool): True to return result as Pandas DataFrame
        dec (int): Rounding to decimal place
    """
    i = 0
    converged = theta + 1
    res = []
    wb = np.concatenate((np.array([bias]), w))
    b = np.array(b)
    while i < max_iter and converged > theta:
        for j in range(len(y)):
            augX = np.concatenate(([1], X[j, :]))
            if y[j] == -1:
                augX = -augX
            yhat = wb.T @ augX
            byhat = (b[j] - yhat) * augX.T
            wb = wb + eta * byhat
            res.append((np.round(yhat, dec), np.round(wb, dec), np.round(y[j] - yhat, dec)))
            i = i + 1
            converged = np.abs(np.sum(byhat))
    return pd.DataFrame(res, columns=["yhat", "weights", "error"]) if as_frame else res