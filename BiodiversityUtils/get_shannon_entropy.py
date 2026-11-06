import numpy as np


def get_shannon_entropy(p):
    return -(p[p > 0] * np.log(p[p > 0])).sum()