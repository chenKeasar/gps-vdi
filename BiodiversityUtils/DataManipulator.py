import numpy as np
from BiodiversityUtils.Taxonomy_utils.get_taxonomic_lists import get_taxonomic_lists

class DataManipulator(object):
    def manipulate(self, df):
        pass

class TrapDataRandomizer(DataManipulator):
    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def manipulate(self, df):
        trap_ids = list(df['parent_image_id'])
        randomized_traps = self.rng.permutation(trap_ids)
        df['parent_image_id'] = randomized_traps
        return df

class TrapOrderRandomizer(DataManipulator):
    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def manipulate(self, df):
        # The length of each list is the dataframe length, and they correspond to the order,
        # real_family, and pseudo-family of the insects (rows).
        df = df.copy()
        orders_list, _, _ = get_taxonomic_lists(df)
        orders_array = np.array(orders_list)
        orders = list(set(orders_list))
        traps = list(set(df['parent_image_id']))
        for _ in range(len(traps)):
            trap1, trap2 = self.rng.choice(traps, size=2, replace=False)
            order = orders[self.rng.randint(len(orders))]
            mask1 = (orders_array == order) & (df['parent_image_id'] == trap1)
            mask2 = (orders_array == order) & (df['parent_image_id'] == trap2)

            idx1 = df.index[mask1]
            idx2 = df.index[mask2]

            df.loc[idx1, 'parent_image_id'] = trap2
            df.loc[idx2, 'parent_image_id'] = trap1

        return df
