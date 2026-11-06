from BiodiversityUtils.biodiversity_estimators.BiodiversityEstimator import BiodiversityEstimator
from BiodiversityUtils.entropy_estimators import KdeEntropyEstimator, KdeLooEntropyEstimator, CombinedKdeEntropyEstimator
from BiodiversityUtils.entropy_estimators import GridEntropyEstimator

from itertools import product


def get_biodiversity_estimators_15062026(parameters, rng, bandwidths=None, entropy_estimators=None, grid_scaler=None, df=None):
    max_repeats = 1000
    sample_size = 50
    engineered_feature_set = parameters['FEATURE_SET']
    # engineered_feature_set.extend(['red_svd_entropy', 'green_svd_entropy', 'blue_svd_entropy'])



    feature_sets = [{'set': engineered_feature_set, 'name': 'engineered'}]
    # {'set': embeddings_feature_set , 'name': 'embeddings'}]
    estimators = []
    for entropy_estimator, feature_set in product(entropy_estimators, feature_sets):
        feature_set_name = feature_set['name']
        name = f"{entropy_estimator['name']}_{feature_set['name']}"
        estimator = entropy_estimator['estimator']
        feature_set = parameters['FEATURE_SET']
        if entropy_estimator['name'] == 'Grid':
            grid = True
            n_features = 5
            bandwidth = parameters['LAMBDA']
            scaler = grid_scaler
        else:
            grid = False
            n_features = 5
            bandwidth = bandwidths[feature_set_name]
            scaler = None

        estimators.append(BiodiversityEstimator(parameters=parameters, estimator=estimator, max_repeats=max_repeats,
                                                sample_size=sample_size, bandwidth=bandwidth,
                                                rng=rng, feature_set=feature_set,
                                                name=name, n_features=n_features, grid=grid, scaler=scaler, df=df))
    return estimators