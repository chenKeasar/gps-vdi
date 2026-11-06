import numpy as np
import pandas as pd
from pathlib import Path

def get_grid_parameters(parameters, df):
    feature_set = parameters['FEATURE_SET']
    grid_parameters = {}
    for feature in feature_set:
        values = df[feature].to_numpy()
        grid_parameters[feature] = {'min': np.min(values), 'max': np.max(values)}
    grid_parameters_df = pd.DataFrame.from_dict(grid_parameters, orient="index")
    grid_parameters_df.index.name = "feature"
    return grid_parameters_df
