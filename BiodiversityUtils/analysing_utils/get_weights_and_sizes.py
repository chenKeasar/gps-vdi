import numpy as np

def get_weights_and_sizes(n_boxes):
    weights = np.sqrt(n_boxes)
    min_n_sqrt = np.sqrt(np.min(n_boxes))
    max_n_sqrt = np.sqrt(np.max(n_boxes))
    sizes = 20 + 200 * (np.sqrt(n_boxes) - min_n_sqrt) / (max_n_sqrt - min_n_sqrt)
    return weights, sizes