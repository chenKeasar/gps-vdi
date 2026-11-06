import numpy as np

def weighted_corr(x, y, weights):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)

    w = weights / weights.sum()

    x_mean = np.sum(w * x)
    y_mean = np.sum(w * y)

    cov_xy = np.sum(w * (x - x_mean) * (y - y_mean))
    var_x = np.sum(w * (x - x_mean) ** 2)
    var_y = np.sum(w * (y - y_mean) ** 2)

    return cov_xy / np.sqrt(var_x * var_y)