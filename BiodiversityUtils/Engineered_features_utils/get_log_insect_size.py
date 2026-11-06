import pandas as pd
import numpy as np

def get_log_insect_size(df):
    sizes_pixels = df['insect_size[pixels]'].to_numpy().reshape(-1, 1) + 1
    resolution = df['pixels_per_inch'].to_numpy().reshape(-1, 1)
    resolutions = resolution / 1200
    resolutions2 = resolutions ** 2
    sizes = sizes_pixels / resolutions2
    log_sizes = np.log(sizes)
    return log_sizes