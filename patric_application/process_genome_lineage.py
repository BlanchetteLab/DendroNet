"""
A preprocessing file for the AMR data using the genome lineage data to construct a phylogenetic tree approximation
"""

import copy
import numpy as np
import pandas as pd
from patric_application.parse_patric_tree import PatricNode, store_tree_and_leaves, load_tree_and_leaves


def recursively_copy_levels(parent_node, df, parent_level_index, levels, level_sets, leaves):
    if parent_level_index is not None and parent_node.level != 'species' and parent_node.level != 'genome_id' and parent_node.level != 'genus':
        parent_node.taxonomy_dict[parent_node.level] = parent_node.name
    # node passed in was the root
    if parent_level_index is None:
        level_index = 0  # value used when parent_node is the root
        for val in level_sets[level_index]:
            new_node = PatricNode(name=val, parent=parent_node, height=parent_node.height+1.0, level=levels[level_index])
            new_node.taxonomy_dict = copy.deepcopy(parent_node.taxonomy_dict)
            parent_node.descendants.append(new_node)
            recursively_copy_levels(new_node,  df, level_index, levels, level_sets, leaves)
    elif parent_level_index < (len(levels)-1):  # deals with internal nodes
        level_index = parent_level_index + 1
        curr_level = levels[level_index]
        descendant_names = set()
        for row in df[df[parent_node.level] == parent_node.name].itertuples():
            if hasattr(row, curr_level) and getattr(row, curr_level) not in descendant_names:
                curr_name = getattr(row, curr_level)
                descendant_names.add(curr_name)
                new_node = PatricNode(name=curr_name, parent=parent_node, height=parent_node.height+1.0, level=curr_level)
                new_node.taxonomy_dict = copy.deepcopy(parent_node.taxonomy_dict)
                parent_node.descendants.append(new_node)
                recursively_copy_levels(new_node, df, level_index, levels, level_sets, leaves)
    else:  # marks leaves
        parent_node.is_leaf = True
        leaves.append(parent_node)


# note that this cannot be called on leaves, it would mark all of them for not having children
def recursively_mark_for_prune(node):
    node.is_present = False
    if node.descendants is not None and len(node.descendants) > 0:
        for child in node.descendants:
            if child.is_present:
                node.is_present = True
    if node.parent is not None:
        recursively_mark_for_prune(node.parent)


# Starts at root, works down. Assumes the root is present.
def recursively_prune(node):
    pruned_descendants = list()
    for child in node.descendants:
        if child.is_present:
            recursively_prune(child)
            pruned_descendants.append(child)
    node.descendants = pruned_descendants


def create_labelled_tree(data_file='genome_lineage', labels_file='clostridium_samples_combined_resistance.csv'):
    df = pd.read_csv(data_file, delimiter='\t', dtype=str)
    df = df[df.kingdom == 'Bacteria']
    # using 'class' as a column name causes problems later when we iterate over the frame, as it is a python keyword
    df = df.rename(columns={'class': 'safe_class'})
    levels = ['kingdom', 'phylum', 'safe_class', 'order', 'family', 'genus', 'species', 'genome_id']
    # filtering out NaNs
    df = df[(df['kingdom'].notnull()) & (df['phylum'].notnull()) & (df['safe_class'].notnull()) &
            (df['order'].notnull()) & (df['family'].notnull()) & (df['genus'].notnull()) & (df['species'].notnull())
            & (df['genome_id'].notnull())]

    kingdoms = list(set(df['kingdom']))
    phylums = list(set(df['phylum']))
    classes = list(set(df['safe_class']))
    orders = list(set(df['order']))
    families = list(set(df['family']))
    genuses = list(set(df['genus']))
    species = list(set(df['species']))
    ids = set(df['genome_id'])
    level_sets = [kingdoms, phylums, classes, orders, families, genuses, species, ids]

    labels_df = pd.read_csv(labels_file, dtype=str)
    label_ids = set(labels_df['ID'])
    init_num_ids = len(label_ids)
    label_ids = ids.intersection(label_ids)
    if len(label_ids) != init_num_ids:
        print('Missing ' + str(init_num_ids - len(label_ids)) + ' label IDs in the lineage, ' + str(len(label_ids))
              + ' examples remaining')

    root_node = PatricNode(name='root', level='root')
    leaves = list()
    recursively_copy_levels(root_node, df, None, levels, level_sets, leaves)

    """
    some of the following code / functions called is all pretty close to the parse_tree_json stuff in the fungi 
    experiment. It's possible some of it could be consolidated if we wanted to generalize the binary tree logic
     to multifurcating trees 
    """

    for leaf in leaves:
        # constructing the one-hot indicators of taxonomic classification
        new_dict = dict()
        for level_key in leaf.taxonomy_dict.keys():
            level_index = levels.index(level_key)
            one_hot = np.zeros(shape=(len(level_sets[level_index])))
            hot_index = level_sets[level_index].index(leaf.taxonomy_dict[level_key])
            one_hot[hot_index] = 1.0
            new_dict[level_key] = one_hot
        leaf.taxonomy_dict = new_dict
        if leaf.name not in label_ids:
            leaf.is_present = False
        # else:
        #     leaf.name = str(leaf.name)  # convert IDs from floats to strings

    for leaf in leaves:
        recursively_mark_for_prune(leaf.parent)

    recursively_prune(root_node)
    leaves = [leaf for leaf in leaves if leaf.is_present]

    return root_node, leaves

# # for testing
if __name__ == "__main__":

    antibiotics = ['ciprofloxacin', 'cloramphenicol', 'cotrimoxazole',
                   'fusidicacid', 'gentamicin', 'rifampin', 'trimethoprim', 'vancomycin', 'betalactam']
    # antibiotics = ['betalactam']

    for ab in antibiotics:
        print('Starting ' + ab)
        labels_file = 'data_files/' + ab + '_firmicutes_samples.csv'
        storage_file = 'patric_tree_storage/' + ab
        labelled_tree, leaves = create_labelled_tree(data_file='data_files/genome_lineage',
                                                     labels_file=labels_file)
        print('done ' + ab)
        store_tree_and_leaves(labelled_tree, leaves, storage_file)
    # new_tree, new_leaves = load_tree_and_leaves('patric_tree_storage/erythromycin')
    print('done')