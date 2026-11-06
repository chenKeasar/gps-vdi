from pathlib import Path
import numpy as np
import pandas as pd
import json

def get_pca_parameters(parameters):
    timestamp = parameters["Timestamp"]
    method = parameters['METHODS']['bioclip_v2_vitl14_cuda_take1']
    method_id = method["method_id"]
    pca_path = Path(method['pca_path'])
    pca_data_path = pca_path / f"global_pca_{method_id}_{timestamp}.npz"
    pca_data = np.load(pca_data_path)

    mean = pca_data["mean"]
    pca_components = pca_data["components"]
    stats = pca_data["stats"].item()
    stats = json.loads(stats)
    low = np.array(stats["p_low"])
    high = np.array(stats["p_high"])

    return mean, pca_components, low, high