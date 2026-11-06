import numpy as np
import pandas as pd

from BiodiversityUtils.BiodiversityParameters import BiodiversityParameters
from BiodiversityUtils.Engineered_features_utils.get_log_insect_size import get_log_insect_size
from BiodiversityUtils.Engineered_features_utils.get_svd_normalized_entropy import get_svd_normalized_entropy



def get_feature_ranges():
    feature_ranges = {}
    feature_ranges['log_insect_size'] = {
            'min': 3.09,
            'max': 12.21,
            'mean': 8.06,
            'std': 1.37,
            'median': 3.07
    }
    feature_ranges['svd_normalized_entropy'] = {
            'min': 0.27118194,
            'max': 0.90954804,
            'mean': 0.7908177,
            'std': 0.04133214,
            'median': 0.7920833
    }
    feature_ranges['svd_grayscale_normalized_entropy'] = {
            'min': 0.12226269,
            'max': 0.6957074,
            'mean': 0.43370426,
            'std': 0.10395446,
            'median': 0.4247735
    }
    feature_ranges['rb_ratio'] = {
            'min': 2.3975465416330883,
            'max': 45.56238955295568,
            'mean': 7.648572934213314,
            'std': 3.680100152253597,
            'median': 6.619996001067888
    }
    feature_ranges['rg_ratio'] = {
            'min': 1.2386624218270248,
            'max': 1.6093218010830037,
            'mean': 1.3743501741841408,
            'std': 0.047599653031716496,
            'median': 1.3721563332174762
    }
    return feature_ranges

def normalize(raw_feature, feature_range):
    feature = raw_feature - feature_range['min']
    feature = feature / (feature_range['max'] - feature_range['min'])
    np.clip(feature, 0, 1, out=feature)
    return feature


def get_engineered_features_entropy(df):
    parameters = BiodiversityParameters()
    parameters = parameters.current
    timestamp = parameters['Timestamp']
    feature_set = parameters['FEATURE_SET'] # currently ['log_insect_size', 'svd_normalized_entropy', 'svd_grayscale_normalized_entropy', 'rb_ratio', 'rg_ratio']

    log_insect_size                  = df['log_insect_size'].to_numpy().reshape(-1, 1) # get_log_insect_size(df)
    svd_normalized_entropy           = df['svd_normalized_entropy'].to_numpy().reshape(-1, 1) # get_svd_normalized_entropy(df, 'svd')
                                                                                              # svd_normalized_entropy = svd_normalized_entropy.reshape(-1, 1)
    svd_grayscale_normalized_entropy = df['svd_grayscale_normalized_entropy'].to_numpy().reshape(-1, 1)  # get_svd_normalized_entropy(df, 'svd_grayscale')
                                                                                                         # svd_grayscale_normalized_entropy = svd_grayscale_normalized_entropy.reshape(-1, 1)
    rb_ratio                         = df['rb_ratio'].to_numpy().reshape(-1, 1)
    rg_ratio                         = df['rg_ratio'].to_numpy().reshape(-1, 1)

    feature_ranges = get_feature_ranges()

    log_insect_size                  = normalize(log_insect_size, feature_ranges['log_insect_size'])
    svd_normalized_entropy           = normalize(svd_normalized_entropy, feature_ranges['svd_normalized_entropy'])
    svd_grayscale_normalized_entropy = normalize(svd_grayscale_normalized_entropy, feature_ranges['svd_grayscale_normalized_entropy'])
    rb_ratio                         = normalize(rb_ratio, feature_ranges['rb_ratio'])
    rg_ratio                         = normalize(rg_ratio, feature_ranges['rg_ratio'])
    features = np.hstack([log_insect_size, svd_normalized_entropy, svd_grayscale_normalized_entropy, rb_ratio, rg_ratio])
    return get_feature_entropy(features, parameters)


def get_feature_entropy(features, parameters):
    gaussian_lambda = parameters['LAMBDA']
    n_bins = parameters['N_BINS']
    grid = get_grid(n_bins)
    descriptors_shape = features.shape
    descriptors_shape = np.array([descriptors_shape[0], 1, descriptors_shape[1]])
    descriptors = features.reshape(descriptors_shape)
    grid_shape = grid.shape
    grid_shape = np.array([1, grid_shape[0], grid_shape[1]])
    grid = grid.reshape(grid_shape)
    diff2 = (grid - descriptors) ** 2
    diff2_sum = np.sum(diff2, 2)
    distances = np.sqrt(diff2_sum)
    gaussians = np.exp(-gaussian_lambda * distances * distances)
    gaussians_sum = np.sum(gaussians.astype('f4'), 0)  # + biodiversity_parameters['EPSILON']
    q = np.sum(gaussians_sum)
    p = gaussians_sum / q
    log_p = np.log(p + parameters['EPSILON'])  # for numerical stability
    entropy = 0 - np.sum(p * log_p)
    return entropy


def get_grid(n_bins):
    grid = []
    x = []
    for i in range(n_bins):
        x.append(np.float16(i / (n_bins - 1)))
    for i in x:
        for j in x:
            for k in x:
                for l in x:
                    for l1 in x:
                        grid.append([i, j, k, l, l1])
    grid = np.array(grid)
    return grid

