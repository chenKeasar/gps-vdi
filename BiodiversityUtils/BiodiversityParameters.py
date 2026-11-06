"""Analysis parameters.

`biodiversity_parameters_15062026_v3_annotated` is the parameter set behind the published
analysis; the notebook imports it as `current_parameters`. Paths are left as None and are
filled in by the caller, so nothing here is tied to a particular machine.
"""


class BiodiversityParameters(object):
    def __init__(self):
        self.current = biodiversity_parameters_15062026_v3_annotated()


def biodiversity_parameters_15062026_v2(annotated=False):
    timestamp = '15062026_v2'
    parameters = {
        'Experiments': None,
        'Timestamp': timestamp,
        'Comment': None,
        'DATA_ROOT': None,
        'EPSILON': 0.00001,
        'LAMBDA': 40,
        'N_BINS': 5,
        'DENSITY_THRESHOLD': 1,
        'FEATURE_SET': ['log_insect_size', 'rb_ratio', 'rg_ratio', 'svd_ratio12', 'svd_ratio01'],
        'FEATURE_SET_DATE': '11/7/2026',
        'ANALYSIS_LEVELS': ['TRAP', 'GROUP', 'EXPERIMENT'],
        'ANNOTATED': annotated,
        'Seed': 0,
    }
    return parameters


def biodiversity_parameters_15062026_v3(annotated=False):
    parameters = biodiversity_parameters_15062026_v2(annotated=annotated)
    parameters['Timestamp'] = '15062026_v3'
    parameters['tarps_path'] = None
    return parameters


def biodiversity_parameters_15062026_v3_annotated():
    return biodiversity_parameters_15062026_v3(annotated=True)
