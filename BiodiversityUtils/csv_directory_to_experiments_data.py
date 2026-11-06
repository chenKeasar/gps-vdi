import pandas as pd
from pathlib import Path


def csv_directory_to_experiments_data(parameters, meta_group_annotator=None):
    experiments_data = {}
    data_root = Path(parameters['DATA_ROOT'])
    experiments_data['csv_files_path'] = data_root / "csv_files"
    csv_files = experiments_data['csv_files_path'].glob("*.csv")
    experiments = []
    dfs = []
    for csv_file in csv_files:
        experiments.append(csv_file.stem)
        dfs.append(pd.read_csv(csv_file))
    df = pd.concat(dfs, ignore_index=True)
    if meta_group_annotator is not None:
        df = meta_group_annotator.annotate(df)
    df["experiment_name"] = df["experiment_name"].str.replace(" ", "_", regex=False) # Replaces white_spaces with _
    n1 = len(df)
    df.dropna(subset=parameters['FEATURE_SET'], inplace=True)
    n2 = len(df)
    print(f"{n1 - n2} insects with NaN values were removed.")
    experiments_data['df'] = df
    experiments_data['experiments'] = experiments
    experiments_data['experiments_string'] = '_'.join(experiments)
    return experiments_data