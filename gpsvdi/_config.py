"""Feature-extraction constants, vendored verbatim from STARdbi's
helpers/biodiversity_config.py so this package has no STARdbi dependency."""

BIODIVERSITY_FEATURE_COLUMNS = ['rb_ratio', 'rg_ratio', 'bg_ratio', 'log_insect_size', 'svd_normalized_entropy', 'svd_ratio01', 'svd_ratio12']
COLOR_RATIO_OFFSET = 1
GRID_COLLAPSE_TRIVIAL_REPEATS = True
GRID_ENGINEERED_COLUMN = 'grid_engineered'
GRID_EPSILON = 1e-05
GRID_FEATURE_SET = ['log_insect_size', 'rb_ratio', 'rg_ratio', 'svd_ratio12', 'svd_ratio01']
GRID_LAMBDA = 40
GRID_MAX_REPEATS = 1000
GRID_N_BINS = 5
GRID_N_FEATURES = 5
GRID_RANDOM_SEED = 42
GRID_SAMPLE_SIZE = 50
PPI_REFERENCE = 1200
SIZE_PIXEL_OFFSET = 1
SVD_MIN_SINGULAR_VALUE = 1.0
SVD_RATIO_DOUBLE_LOG = True
SVD_RATIO_PAIRS = {'svd_ratio01': (0, 1), 'svd_ratio12': (1, 2)}
