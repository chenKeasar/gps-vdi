# The column holding the full taxonomic path (order__suborder__infraorder__superfamily__
# family) has been called family_name; it is being renamed to something less misleading.
# Accept any of these so the deposited table and this code can be updated independently.
HIERARCHY_COLUMNS = ("lineage", "taxonomic_hierarchy", "family_name")


def hierarchy_column(df):
    for name in HIERARCHY_COLUMNS:
        if name in df.columns:
            return name
    raise KeyError("no taxonomic hierarchy column; expected one of "
                   + ", ".join(HIERARCHY_COLUMNS))


def get_taxonomic_lists(trap_df):
    orders = []
    families = []
    real_families = []

    names = list(trap_df[hierarchy_column(trap_df)])
    for name in names:
        words = name.split('__')
        orders.append(words[0])
        families.append(words[4])
        if words[3] != words[4]:
            real_families.append(words[4])
        else:
            real_families.append("Unknown")
    return orders, real_families, families

def add_order_and_family(df):
    orders, real_families, families = get_taxonomic_lists(df)
    df["order"] = orders
    df["family"] = real_families
