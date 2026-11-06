from BiodiversityUtils.BiodiversityParameters import BiodiversityParameters
from BiodiversityUtils.Embeddings_utils.Embedding_utils import get_single_path
from BiodiversityUtils.Engineered_features_utils.get_engineered_features_entropy import get_feature_entropy
from BiodiversityUtils.Embeddings_utils.get_pca_parameters import get_pca_parameters
from pathlib import Path
import pandas as pd
import numpy as np
import json



def get_embeddings_entropy(parameters, trap_df, data_root=None):
    mean, pca_components, low, high = get_pca_parameters(parameters)
    """timestamp     = parameters["Timestamp"]
    method        = parameters['METHODS']['bioclip_v2_vitl14_cuda_take1']
    method_id     = method["method_id"]
    pca_path      = Path(method['pca_path'])
    pca_data_path = pca_path / f"global_pca_{method_id}_{timestamp}.npz"
    pca_data      = np.load(pca_data_path)

    mean           = pca_data["mean"]
    pca_components = pca_data["components"]
    stats          = pca_data["stats"].item()
    stats          = json.loads(stats)
    low            = np.array(stats["p_low"])
    high           = np.array(stats["p_high"])"""

    trap_id              = str(trap_df.iloc[0]['parent_image_id'].item()).zfill(5)
    # Embeddings are looked up directly under data_root rather than in a
    # per-dataset subfolder; the commented line below is the older layout.
    # trap_embeddings_path = (Path(data_root) / 'for_embedding').rglob(f"{trap_id}_*.npz")
    trap_embeddings_path = Path(data_root).rglob(f"{trap_id}_*.npz")
    trap_embeddings_path = get_single_path(trap_embeddings_path, description=trap_id)
    embedding_data       = np.load(trap_embeddings_path)
    trap_embeddings      = embedding_data["embedding"]

    transformed_embeddings = (trap_embeddings - mean) @ pca_components.T
    transformed_embeddings = transformed_embeddings[:,1:6]
    transformed_embeddings = (transformed_embeddings - low[1:6]) / (high[1:6] - low[1:6])

    return get_feature_entropy(transformed_embeddings, parameters)