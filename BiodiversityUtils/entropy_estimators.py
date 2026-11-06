from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from itertools import product


# estimators.py
EPSILON = 1e-12

@dataclass
class EntropyResult:
    entropy: float
    log_likelihood: float
    bandwidth: float

class EntropyEstimator(ABC):
    name = "abstract"

    @abstractmethod
    def estimate_entropy(self, x_embeddings: np.ndarray) -> EntropyResult:
        """
            Parameters
            ----------
            x_embeddings : np.ndarray
                Corresponds to X in standard mathematical notation:
                a matrix of samples, shape (n_samples, n_features).
        """
        pass



class KdeEntropyEstimator(EntropyEstimator):
    name = "kde"

    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth

    def estimate_entropy(self, x_embeddings: np.ndarray) -> EntropyResult:
        """
                H ≈ -mean(log p_hat(x_i)); includes self-contribution.
                Returns:
                    entropy ≈ -E[log p̂(x)]
                    log_likelihood = sum log p̂(x_i)
        """
        kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth).fit(x_embeddings)
        log_p = kde.score_samples(x_embeddings)
        entropy = -np.mean(log_p)
        log_likelihood = np.sum(log_p)
        return EntropyResult(
            entropy=entropy,
            log_likelihood=log_likelihood,
            bandwidth=self.bandwidth
        )


class KdeLooEntropyEstimator(EntropyEstimator):
        name = "kde_loo"

        def __init__(self, bandwidth=None):
            self.bandwidth = bandwidth

        def estimate_entropy(self, X: np.ndarray) -> EntropyResult:
            """Leave-one-out KDE entropy using analytic self-term subtraction."""
            bandwidth = self.bandwidth
            n, d = X.shape
            kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X)

            log_p = kde.score_samples(X)
            p = np.exp(log_p)

            # Gaussian kernel peak at zero distance
            K0 = 1.0 / ((2 * np.pi) ** (d / 2) * bandwidth ** d)

            # remove self-contribution and renormalize from n to n-1
            p_loo = (n * p - K0) / (n - 1)
            p_loo = np.maximum(p_loo, EPSILON)
            log_p_loo = np.log(p_loo)
            entropy = float(-np.mean(log_p_loo))
            log_likelihood = np.sum(log_p_loo)
            return EntropyResult(
                entropy=entropy,
                log_likelihood=log_likelihood,
                bandwidth=self.bandwidth
            )




class CombinedKdeEntropyEstimator(EntropyEstimator):
    name = "combined"

    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth
        self.kde = KdeEntropyEstimator(bandwidth)
        self.loo_kde = KdeLooEntropyEstimator(bandwidth)

    def estimate_entropy(self, X: np.ndarray) -> EntropyResult:
        self.loo_kde.bandwidth = self.bandwidth
        self.kde.bandwidth = self.bandwidth
        entropy_results1 = self.loo_kde.estimate_entropy(X)
        entropy_results2 = self.kde.estimate_entropy(X)
        entropy = (entropy_results1.entropy + entropy_results2.entropy) / 2.0
        log_likelihood = (entropy_results1.log_likelihood + entropy_results2.log_likelihood) / 2.0
        return EntropyResult(
            entropy=entropy,
            log_likelihood=log_likelihood,
            bandwidth=self.bandwidth
        )


def kde_bandwidth_cv_results(
    x_embeddings: np.ndarray,
    bandwidths=None,
    cv: int = 5,
):
    if bandwidths is None:
        bandwidths = np.logspace(-1, 1, 100)

    grid = GridSearchCV(
        KernelDensity(kernel="gaussian"),
        {"bandwidth": bandwidths},
        cv=cv,
    )

    grid.fit(x_embeddings)

    return grid.best_params_["bandwidth"], grid.cv_results_

class OneDGridEntropyEstimator(EntropyEstimator):
    name = "1D grid"
    def __init__(self, bandwidth=None, grid_min=-5, grid_max=5, grid_size=2000):
        self.bandwidth = bandwidth
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.grid_size = grid_size
        self.H_differential_approx = -1

    def estimate_entropy(self, X: np.ndarray) -> float:
        
        # Discretized KDE entropy:
        # splat Gaussian kernels onto a 1D grid, normalize grid mass,
        # then compute Shannon entropy over grid cells.
        
        x = X[:, 0]
        grid = np.linspace(self.grid_min, self.grid_max, self.grid_size)
        dx = grid[1] - grid[0]

        density = np.zeros_like(grid)

        for xi in x:
            density += np.exp(-0.5 * ((grid - xi) / self.bandwidth) ** 2)

        # Include Gaussian normalization if you want true density;
        # for discrete entropy after normalization it cancels.
        density = density / density.sum()

        self.H_discrete = -np.sum(density * np.log(np.maximum(density, EPSILON)))

        # Optional: approximate differential entropy correction
        H_differential_approx = self.H_discrete + np.log(dx)

        return H_differential_approx

class GridEntropyEstimator(EntropyEstimator):
    name = "GridEntropy"
    def __init__(self, bandwidth=None, n_bins=None, n_features=None):
        self.bandwidth = bandwidth
        self.n_bins = n_bins
        self.n_features = n_features
        self.grid = self.get_grid()
        print(self.grid.shape)

    def estimate_entropy(self, features: np.ndarray) -> EntropyResult:
            descriptors_shape = features.shape
            descriptors_shape = np.array([descriptors_shape[0], 1, descriptors_shape[1]])
            descriptors = features.reshape(descriptors_shape)
            grid_shape = self.grid.shape
            grid_shape = np.array([1, grid_shape[0], grid_shape[1]])
            grid = self.grid.reshape(grid_shape)
            diff2 = (grid - descriptors) ** 2
            diff2_sum = np.sum(diff2, 2)
            distances = np.sqrt(diff2_sum)
            gaussians = np.exp(-self.bandwidth * distances * distances)
            gaussians_sum = np.sum(gaussians.astype('f4'), 0)  # + biodiversity_parameters['EPSILON']
            q = np.sum(gaussians_sum)
            p = gaussians_sum / q
            log_p = np.log(p + 0.00001)
            entropy = 0 - np.sum(p * log_p)
            log_likelihood = np.sum(log_p)
            return EntropyResult(
                entropy=entropy,
                log_likelihood=log_likelihood,
                bandwidth=self.bandwidth
            )

    def get_grid(self):
        x = np.linspace(0, 1, self.n_bins)
        return np.array(list(product(x, repeat=self.n_features)))




