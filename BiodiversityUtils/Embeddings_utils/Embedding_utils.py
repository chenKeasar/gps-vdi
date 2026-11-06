from sklearn.decomposition import PCA
import json
from datetime import datetime
import pandas as pd
from pathlib import Path


def check_data(df):
    bad = df.groupby("parent_image_id")["image_meta_group"].nunique()
    bad = bad[bad > 1]
    assert bad.empty, f"Some parent_image_id map to multiple groups:\n{bad.head(20)}"

def get_trap_group_map(data_root, experiment):
    csv_file = data_root / experiment / f'data/csv_files/{experiment}.csv'
    if not csv_file.exists():
        csv_file = data_root / f"csv_files/{experiment}.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"File {csv_file} not found.")
    df = pd.read_csv(csv_file)
    check_data(df)
    df = df[["parent_image_id", "image_meta_group"]].drop_duplicates("parent_image_id")
    trap_group_map = dict(
        zip(df["parent_image_id"], df["image_meta_group"])
    )
    return trap_group_map

def load_model(method):
    import birder
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = method['embedding_model']
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    net, model_info = birder.load_pretrained_model(model, inference=True)

    net.to(device).eval()

    size = birder.get_size_from_signature(model_info.signature)
    transform = birder.classification_transform(size, model_info.rgb_stats)

    return net, transform, device

def build_trap_metadata(experiment, trap_id, group_id, names, X, method):
    metadata = build_global_metadata(names, X, method)
    metadata["dataset"].update({
        "experiment": experiment,
        "trap_id": trap_id,
        "group_id": group_id
    })
    return metadata


def build_global_metadata(names, X, method):
    return {
        "dataset": {
            "n_images": len(names),
            "embedding_dim": int(X.shape[1]),
            "created": datetime.now().isoformat()
        },
        "method": method
    }

def batched(iterable, batch_size):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def get_embeddings_batch(image_paths, net, transform, device):
    from PIL import Image
    import torch
    tensors = []
    names = []

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to open image '{image_path}'"
            ) from e
        try:
            x = transform(image)  # (C,H,W), CPU
        except Exception as e:
            raise RuntimeError(
                f"Failed to transform image '{image_path}'"
            )
        tensors.append(x)
        names.append(image_path.name)

    try:
        batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)  # (B,C,H,W)
    except Exception as e:
        batch_names = [p.name for p in image_paths]
        raise RuntimeError(
            f"Failed to embed batch of {len(image_paths)} images. "
            f"First image: '{image_paths[0]}'. "
            f"Last image: '{image_paths[-1]}'."
        ) from e

    embedding = net.embedding(batch)   # (B,1024)
    embedding = embedding.detach().cpu().numpy().astype(np.float32)

    return names, embedding

def print_embeddings_and_metadata(embeddings_path=None, embeddings=None, names=None, metadata=None):
    metadata_json = json.dumps(metadata, sort_keys=True)
    # Save compressed binary file
    np.savez_compressed(embeddings_path,
                        embeddings_path=embeddings_path,
                        image_name=names,
                        embedding=embeddings,
                        metadata_json=metadata_json)
                # save metadata as json
    data = {
            "metadata": metadata,
            "embedding_file": str(embeddings_path),
            "images": names.tolist(),
    }
    json_path = embeddings_path.with_suffix(".json")
    with json_path.open("w") as f:
        json.dump(data, f, sort_keys=True)


def calculate_embeddings_for_experiment(parameters=None, method=None, experiment=None, net=None, transform=None, device=None):
    import torch
    from tqdm import tqdm
    batch_size = method.get("batch_size", 32)
    data_root = parameters['DATA_ROOT']

    with torch.inference_mode():
        # Assumes data is arranged in experiments/traps/boxes hierarchy
        # Written in February 2026. Future versions should be more general
        print('Extracting from ' + experiment)
        trap_group_map = get_trap_group_map(data_root, experiment)
        data_path = data_root / experiment / 'data' / 'boxes'
        if not data_path.exists():
            data_path = data_root / 'boxes'
        if not data_path.exists():
            raise FileNotFoundError(f"File {data_path} not found.")
        for trap_path in data_path.glob('?????'):
            if not trap_path.is_dir():
                continue
            trap_id = trap_path.name
            if parameters['GROUPS']:
                group_id = trap_group_map[int(trap_id)]
            else:
                group_id = 0
            group_id = f"{group_id:05d}"

            embeddings_path = trap_path.parent /  f"{trap_id}_{group_id}_{method['method_id']}_{parameters['Timestamp']}.npz"
            if embeddings_path.exists():
                continue

            print('          Extracting from ' + trap_id)

            names, embeddings = [], []
            image_paths = sorted([p for p in trap_path.glob("*") if p.is_file()])
            for image_batch in tqdm(batched(image_paths, batch_size)):
                batch_names, batch_embeddings = get_embeddings_batch(image_batch, net, transform, device)
                names.extend(batch_names)
                embeddings.append(batch_embeddings)
            if not embeddings:
                continue

            X = np.vstack(embeddings).astype(np.float32)
            names = np.array(names, dtype=str)
            metadata = build_trap_metadata(experiment, trap_id, group_id, names, X, method)
            print_embeddings_and_metadata(embeddings_path=embeddings_path, embeddings=X, names=names, metadata=metadata)

""" This code is now a stand alone program 
def load_all_embeddings(data_root, npz_files, method_id, method):
    mats = []
    names = []
    for f in tqdm(npz_files):
        with np.load(f) as z:
            mats.append(z["embedding"].astype(np.float32, copy=False))
            names.extend(z["image_name"])
    embeddings_data = np.vstack(mats)
    embeddings_path = data_root / 'embeddings' / f"embeddings_{method_id}_{method["timestamp"]}.npz"
    embeddings_json_path = embeddings_path.with_suffix(".json")
    np.savez_compressed(embeddings_path, embedding=embeddings_data)
    return embeddings_data, names  # (N, 1024), N
    """

import numpy as np


def compute_embedding_range_stats(X, lower_percentile=1, upper_percentile=99):
    """
    Compute min/max and percentile-based range statistics for an embedding matrix.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, embedding_dim)
    lower_percentile : float
    upper_percentile : float

    Returns
    -------
    dict
        Statistics useful for normalization and later analysis
    """

    stats = {
        "min": X.min(axis=0).astype(np.float32),
        "max": X.max(axis=0).astype(np.float32),
        "p_low": np.percentile(X, lower_percentile, axis=0).astype(np.float32),
        "p_high": np.percentile(X, upper_percentile, axis=0).astype(np.float32),
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
    }

    def stats_to_jsonable(stats):
        return {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in stats.items()
        }
    return stats, stats_to_jsonable(stats)

def calc_pca_of_of_a_dataset(parameters=None):
    method = parameters['METHODS']['bioclip_v2_vitl14_cuda_take1']
    data_root = parameters['DATA_ROOT']
    timestamp = parameters["Timestamp"]
    method_id = method["method_id"]
    pca_size = method["pca_size"]
    pca_path = Path(method["pca_path"]) / f"global_pca_{method_id}_{timestamp}.npz"
    embeddings_path = pca_path.parent / f"embeddings_{method_id}_{timestamp}.npz"
    pca_json_path = pca_path.with_suffix(".json")
    with open(pca_path.parent / "names.txt", "r", encoding="utf-8") as f:
        names = [line.rstrip("\n") for line in f]
    data = np.load(embeddings_path)
    pca = PCA(n_components=method["pca_size"], svd_solver="full", random_state=0)
    embeddings = data["embedding"]
    pca.fit(embeddings)
    stats, jsonable_stats = compute_embedding_range_stats(pca.transform(embeddings))
    metadata = build_global_metadata(names, embeddings, method)
    metadata_json = json.dumps(metadata, sort_keys=True)
    stats_json = json.dumps(jsonable_stats, sort_keys=True)
    np.savez_compressed(
        pca_path,
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),  # (pca_size, 1024)
        explained_variance=pca.explained_variance_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        metadata_json=metadata_json,
        stats=stats_json
    )

    data = {
        "pca_path": str(pca_path),
        "mean": pca.mean_.tolist(),
        "components": pca.components_.tolist(),
        "explained_variance": pca.explained_variance_.astype(np.float32).tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32).tolist(),
        "metadata": metadata,
        "stats": stats_json
    }

    with open(str(pca_json_path), "w") as f:
        json.dump(data, f, indent=4)



def fit_pca_reducers(X: np.ndarray, ks=(3,4,5), out_dir=Path(".."), prefix="pca"):
    """
    X: shape (N, 1024)
    Saves one PCA model per k into out_dir, returns dict{k: pca_model}.

    Use it like
    X = np.vstack(df["embedding"].values).astype("float32")  # (N,1024)
    fit_pca_reducers(X, ks=(3,4,5), out_dir=data_root/"reducers")
    """
    import joblib
    out_dir.mkdir(parents=True, exist_ok=True)
    reducers = {}
    for k in ks:
        pca = PCA(n_components=k, svd_solver="auto", random_state=0)
        pca.fit(X)
        reducers[k] = pca
        joblib.dump(pca, out_dir / f"{prefix}_{k}d.joblib")
    return reducers

def load_reducer(path: Path):
    import joblib
    return joblib.load(path)

def reduce_embedding(embedding_1024: np.ndarray, reducer) -> np.ndarray:
    """
    embedding_1024: shape (1024,) or (1,1024)
    returns shape (k,)
    """
    x = np.asarray(embedding_1024)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = reducer.transform(x)
    return y.ravel()


# AI generated method. making sure that a
# generator object Path.rglob
# provides exactly one Path object
def get_single_path(iterator, description=None):
    it = iter(iterator)
    try:
        first = next(it)
    except StopIteration:
        raise FileNotFoundError(f"No {description} found")

    try:
        next(it)
        raise ValueError(f"Multiple {description}s found")
    except StopIteration:
        return first


# loading code. needs to be part of some function
# pca3 = load_reducer(Path("reducers") / "pca_3d.joblib") # or any other "k"
# y3 = reduce_embedding(embedding, pca3)   # y3 has shape (3,)




