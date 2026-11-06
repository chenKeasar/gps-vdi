from abc import ABC, abstractmethod

from numpy.ma.extras import row_stack
from sklearn.preprocessing import StandardScaler as Scaler
# from sklearn.preprocessing import RobustScaler as Scaler

from BiodiversityUtils.entropy_estimators import EntropyEstimator, kde_bandwidth_cv_results, EntropyResult, \
    CombinedKdeEntropyEstimator
from BiodiversityUtils.entropy_estimators import KdeEntropyEstimator, KdeLooEntropyEstimator
from BiodiversityUtils.entropy_estimators import GridEntropyEstimator
from BiodiversityUtils.biodiversity_estimators.GridScaler import GridScaler


import numpy as np
import pandas as pd

class BiodiversityEstimator(EntropyEstimator):
    def __init__(self,
                 parameters=None,
                 estimator=None,
                 max_repeats=None,
                 sample_size=None,
                 bandwidth=None,
                 rng=None,
                 name=None,
                 feature_set=None,
                 n_features=None,
                 grid=None,
                 scaler=None,
                 df=None):
        self.parameters = parameters
        self.feature_set = feature_set
        self.name = name
        if df is None:
            df = pd.read_csv(parameters['merged_csv_file_path'])
        self.x_embeddings = self.extract_features(df)
        if grid:
            print(GridScaler)
            print(type(GridScaler))
            if scaler is None:
                self.scaler = GridScaler(x_embeddings=self.x_embeddings)
            else:
                self.scaler = scaler
        else:
            self.scaler = Scaler().fit(self.x_embeddings)
        self.x_embeddings = self.scaler.transform(self.x_embeddings)
        self.rng = np.random.RandomState(42)
        self.max_repeats = max_repeats
        self.sample_size = sample_size
        self.rng = rng
        self.n_features = n_features
        self.bandwidth = bandwidth
        self.estimator = estimator
        self.estimator.bandwidth = self.bandwidth


    def get_name(self):
        return self.name

    def estimate_entropy(self, df):
        x_embeddings = self.extract_features(df)
        x_embeddings = x_embeddings[:,0:9]
        x_embeddings = self.scaler.transform(x_embeddings)
        n_repeats = self.max_repeats
        repeats = []
        # Chen Debug
        # n_repeats = 1
        # End of chen
        for _ in range(n_repeats):
            n = x_embeddings.shape[0]
            k = min( self.sample_size, n)
            row_indices = self.rng.choice(n, size=k, replace=False)
            column_indices = self.rng.permutation(x_embeddings.shape[1])
            sample = x_embeddings[row_indices][:, column_indices[0:self.n_features]]
            # Chen Debug
            # sample = x_embeddings
            # End of chen
            repeats.append(self.estimator.estimate_entropy(sample))
        results = EntropyResult(entropy=float(np.mean(np.array([r.entropy for r in repeats]))),
                                log_likelihood=float(np.mean(np.array([r.log_likelihood for r in repeats]))),
                                bandwidth=self.bandwidth)
        return results

    def extract_features(self, df):
        df_features = df[self.feature_set]
        return df_features.to_numpy(dtype=np.float64)

    def get_bandwidth(self):
        x_embeddings = self.x_embeddings
        best_bandwidth = kde_bandwidth_cv_results(x_embeddings)
        return best_bandwidth



class EngineeredFeaturesKdeBiodiversityEstimator(BiodiversityEstimator):
    name = "kde_engineered_features"
    def __init__(self, parameters=None, max_repeats=None, sample_size=None, bandwidth=None, rng=None):
        super().__init__(parameters=parameters,
                         estimator=KdeEntropyEstimator(bandwidth=bandwidth),
                         bandwidth=bandwidth,
                         max_repeats=max_repeats,
                         sample_size=sample_size,
                         rng=rng)

class EngineeredFeaturesKdeLooBiodiversityEstimator(BiodiversityEstimator):
    name = "kde_loo_engineered_features"
    def __init__(self, parameters=None, max_repeats=None, sample_size=None, bandwidth=None, rng=None):
        super().__init__(parameters=parameters,
                         estimator=KdeLooEntropyEstimator(bandwidth=bandwidth),
                         bandwidth=bandwidth,
                         max_repeats=max_repeats,
                         sample_size=sample_size,
                         rng=rng)

class EngineeredFeaturesCombinedBiodiversityEstimator(BiodiversityEstimator):
    name = "combined_engineered_features"
    def __init__(self, parameters=None, max_repeats=None, sample_size=None, bandwidth=None, rng=None):
        super().__init__(parameters=parameters,
                         estimator=CombinedKdeEntropyEstimator(bandwidth=bandwidth),
                         bandwidth=bandwidth,
                         max_repeats=max_repeats,
                         sample_size=sample_size,
                         rng=rng)

class EngineeredFeaturesGridBiodiversityEstimator(BiodiversityEstimator):
    name = "grid"
    def __init__(self, parameters=None, max_repeats=None, sample_size=None, n_features=None, rng=None):
        super().__init__(parameters=parameters,
                         estimator=GridEntropyEstimator(bandwidth=parameters['LAMBDA'], n_features=n_features),
                         bandwidth=parameters['LAMBDA'],
                         max_repeats=max_repeats,
                         sample_size=sample_size,
                         rng=rng,
                         grid=True)













