import pandas as pd
import numpy as np
import ast

from BiodiversityUtils.get_shannon_entropy import get_shannon_entropy

def to_numeric_array(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)

    return np.asarray(value, dtype=float)

"""def get_svd_normalized_entropy(df, in_header):
    svd_column = df[in_header]
    values = []
    for key in svd_column.keys():
        svd = np.array(ast.literal_eval(svd_column[key]), dtype=float)
        # svd = np.array(svd_column[key])
        e = get_shannon_entropy(svd)
        if (np.log(len(svd))) <= 0:
            raise Exception('This is weird')
        value = e / np.log(len(svd))
        value = value.item()
        values.append(value)
    values = np.array(values).reshape(-1, 1)
    return values"""

def get_svd_normalized_entropy(df, in_header):
    svd_column = df[in_header]
    values = []

    for key in svd_column.keys():
        svd = to_numeric_array(svd_column[key])

        # If these are singular values, normalize before Shannon entropy
        svd = svd / svd.sum()

        e = get_shannon_entropy(svd)
        values.append(e)

    return np.array(values)

