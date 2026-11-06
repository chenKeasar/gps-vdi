from pathlib import Path

import numpy as np
import pandas as pd
import os
from functools import reduce

from BiodiversityUtils.Taxonomy_utils.get_shannon_index_and_histogram import get_shannon_index_and_histogram
from BiodiversityUtils.Taxonomy_utils.get_taxonomic_lists import get_taxonomic_lists
from BiodiversityUtils.get_embeddings_entropy import get_embeddings_entropy


def analyse_trap_by_taxonomy(trap_df, trap_name, trap_id, trap_group, n_boxes, taxa_list=None, taxonomic_level=None):
    trap_df = trap_df[trap_df['parent_image_id'] == trap_id]
    taxa = trap_df[taxonomic_level]
    taxa_df, _ = get_shannon_index_and_histogram(taxa=taxa, taxa_list=taxa_list, taxonomic_level=taxonomic_level,
                                              trap_id=trap_id, trap_name=trap_name, trap_group=trap_group,
                                              n_boxes=n_boxes)

    return taxa_df

def analyse_unit(unit_df, parameters, estimators, unit='TRAP', taxa_list=None, taxonomic_level=None, debug=False):
    data_root = parameters['DATA_ROOT']
    trap_id = unit_df.iloc[0]['parent_image_id']
    trap_group = unit_df.iloc[0]['image_meta_group']
    trap_group_name = unit_df.iloc[0]['meta_group_name']
    trap_name = unit_df.iloc[0]['parent_image_file_name']
    experiment = unit_df.iloc[0]['experiment_name']
    n_boxes = unit_df.shape[0]

    if (not taxonomic_level is None) and unit == 'TRAP':
        df = analyse_trap_by_taxonomy(unit_df, trap_name, trap_id, trap_group, n_boxes,
                                      taxa_list=taxa_list, taxonomic_level=taxonomic_level)
    else:
        df = pd.DataFrame(columns=['experiment', 'trap_name', 'trap_id',  'trap_group', 'trap_group_name', 'n_boxes'])
        df.loc[0] = [experiment, trap_name, trap_id, trap_group, trap_group_name, n_boxes]

    for estimator in estimators:
        print(f" {estimator.get_name()}")
        entropy_estimation = estimator.estimate_entropy(unit_df)
        df[estimator.get_name()] = float(entropy_estimation.entropy)
    return df


def analyse_by_traps(df, parameters, estimators, taxa_list=None, taxonomic_level=None, seed=None, tag=None, debug=False):
    print('Analyzing traps')
    traps = df.groupby('parent_image_id')
    dfs: list[pd.DataFrame] = []
    for trap_id, trap_df in traps:
        print(f"Processing {trap_id} of size {len(trap_df)}")
        trap_row = analyse_unit(trap_df, parameters, estimators, taxa_list=taxa_list, taxonomic_level=taxonomic_level, debug=debug)
        dfs.append(trap_row)
    out_df = pd.concat(dfs)
    results_file = get_results_file(parameters, taxonomic_level=taxonomic_level, level='trap', seed=seed, tag=tag)
    out_df.to_csv(results_file, index=False)

def analyse_by_groups(df, parameters, estimators, taxa_list=None, taxonomic_level=None, seed=None, tag=None, debug=False):
    print('Analyzing groups')
    groups = df.groupby('image_meta_group')
    dfs: list[pd.DataFrame] = []
    for group_id, group_df in groups:
        print(f"Processing {group_id} of size {len(group_df)}")
        trap_row = analyse_unit(group_df, parameters, estimators, taxa_list=taxa_list, taxonomic_level=taxonomic_level, debug=debug)
        dfs.append(trap_row)
    out_df = pd.concat(dfs)
    results_file = get_results_file(parameters, taxonomic_level=taxonomic_level,level='group', seed=seed, tag=tag)
    out_df.to_csv(results_file, index=False)


def analyse_by_experiment(df, parameters, estimators, taxa_list=None, taxonomic_level=None, seed=None, tag=None, debug=False):
    print('Analyzing experiments')
    experiments = df.groupby('experiment_name')
    dfs: list[pd.DataFrame] = []
    for experiment_name, experiment_df in experiments:
        print(f"Processing {experiment_name} of size {len(experiment_df)}")
        trap_row = analyse_unit(experiment_df, parameters, estimators, taxa_list=taxa_list, taxonomic_level=taxonomic_level, debug=debug)
        dfs.append(trap_row)
    out_df = pd.concat(dfs)
    results_file = get_results_file(parameters, taxonomic_level=taxonomic_level, level='experiment', seed=seed, tag=tag)
    out_df.to_csv(results_file, index=False)


def biodiversity_estimation(parameters=None, estimators=None, data_manipulator=None,
                            debug=False, df=None, taxonomic_level=None, seed=None, tag=None):
    if estimators is None:
        estimators = []
    timestamp = parameters['Timestamp']
    data_root = parameters['DATA_ROOT']
    print(data_root)
    if seed is None:
        seed = parameters['Seed']

    if df is None:
        df = pd.read_csv(parameters['merged_csv_file_path'])

    if not data_manipulator is None:
        df = data_manipulator.manipulate(df)

    if not taxonomic_level is None:
        df = df[df[taxonomic_level] != 'Unknown']
        taxa_list = list(set(df[taxonomic_level]))
        taxa_list.sort()
    else:
        taxa_list = None


    analysis_levels = parameters['ANALYSIS_LEVELS']

    if 'TRAP' in analysis_levels:
        analyse_by_traps(df, parameters, estimators, taxa_list=taxa_list, taxonomic_level=taxonomic_level,
                         seed=seed, tag=tag, debug=debug)
    if 'GROUP' in analysis_levels:
        analyse_by_groups(df, parameters, estimators, taxa_list=taxa_list, taxonomic_level=taxonomic_level,
                          seed=seed, tag=tag, debug=debug)
    if 'EXPERIMENT' in analysis_levels:
        analyse_by_experiment(df, parameters, estimators, taxa_list=taxa_list, taxonomic_level=taxonomic_level,
                              seed=seed, tag=tag, debug=debug)


def experiments_string(parameters):
    experiments_string = '_'.join(parameters['Experiments'])
    return experiments_string

def get_results_file(parameters, taxonomic_level=None, level=None, seed=None, tag=None):
    if seed is None:
        seed = parameters['Seed']
    results_path = parameters['Results_path']
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    timestamp = parameters['Timestamp']
    if taxonomic_level is None:
        if tag is None:
            return results_path / f"biodiversity_per_{level}_{experiments_string(parameters)}_{timestamp}_{seed}.csv"
        else:
            return results_path / f"biodiversity_per_{level}_{experiments_string(parameters)}_{timestamp}_{seed}_{tag}.csv"
    else:
        if tag is None:
            return results_path / f"biodiversity_per_{level}_{taxonomic_level}_{experiments_string(parameters)}_{timestamp}_{seed}.csv"
        else:
            return results_path / f"biodiversity_per_{level}_{taxonomic_level}_{experiments_string(parameters)}_{timestamp}_{seed}_{tag}.csv"
