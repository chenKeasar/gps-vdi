import numpy as np

class GridScaler():
    def __init__(self, x_embeddings=None, grid_parameters=None):
        if not x_embeddings is None:
            self.x_embeddings = x_embeddings
            self.q_min = np.percentile(self.x_embeddings, 1, axis=0)
            self.q_max = np.percentile(self.x_embeddings, 99, axis=0)
            self.q_range = self.q_max - self.q_min
            self.min = np.min(self.x_embeddings, axis=0)
            self.max = np.max(self.x_embeddings, axis=0)
        else:
            self.min = grid_parameters["min"].to_numpy(dtype=float)
            self.max = grid_parameters["max"].to_numpy(dtype=float)

    def transform(self, x_embeddings):
        transformed = (x_embeddings - self.min) / (self.max - self.min)
        return transformed