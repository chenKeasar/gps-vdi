from collections import Counter
import numpy as np
import pandas as pd
from BiodiversityUtils.get_shannon_entropy import get_shannon_entropy

def get_shannon_index_and_histogram(taxa=None, taxa_list=None, taxonomic_level=None,
                                    trap_id=None, trap_name=None, trap_group=None, n_boxes=None):
    counts = Counter(taxa)
    hist_list = [counts.get(taxon, 0) for taxon in taxa_list]
    hist_vector = np.array(hist_list)
    if hist_vector.sum() == 0:
        entropy = -1
    else:
        frequencies = hist_vector / hist_vector.sum()
        entropy = get_shannon_entropy(frequencies)
    row_as_dict = None
    if trap_name is None:
        row_as_dict = {'trap_id': trap_id, 'level': taxonomic_level, f"entropy({taxonomic_level})": entropy}
    if n_boxes is not None:
        row_as_dict = {'trap_name': trap_name, 'trap_id': trap_id, 'trap_group': trap_group, 'n_boxes':n_boxes, 'level': taxonomic_level, f"entropy({taxonomic_level})": entropy}
    for taxon in taxa_list:
        row_as_dict[taxon] = counts.get(taxon, 0)

    df = pd.DataFrame([row_as_dict])

    return df, row_as_dict

"""def get_shannon_index_and_histogram(taxons, taxa_list):
    counts = Counter(taxons)
    hist = {taxon: counts.get(taxon, 0) for taxon in taxa_list}
    hist_vector = np.array([hist[t] for t in taxa_list])
    if (hist_vector.sum() == 0):
        entropy = -1
        hist = None
    else:
        frequencies = hist_vector / hist_vector.sum()
        entropy = get_shannon_entropy(frequencies)
    return hist, entropy"""