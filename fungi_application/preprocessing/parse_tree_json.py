import os
import copy
import json
import pandas as pd

"""
This file is for transforming the phylo_tree.json file and using it to create a binary phylogenetic
tree usable as input to the ML programs. 
"""

PLANT_TRAITS = ['Gymnosperms', 'Angiosperms', 'Stem', 'Leaves', 'Roots']

data_leaves = list()
tree_leaves = list()

"""
A simple class to hold the data from the fungus tree, this was eventually made redundant and could be replaced
"""
class FungiNode:
    def __init__(self, name=None, is_leaf=False, parent=None, height=1, features=None, target=None):
        self.name = name
        self.is_leaf = is_leaf
        self.parent = parent
        self.height = height
        if features is None:
            self.x = dict()
        else:
            self.x = features
        self.lifestyles = None
        self.y = target
        self.left = None
        self.right = None
        self.is_present = True
        self.placement_vector = list()


def load_tree(filepath):
    # on the off chance we call this multiple times in one session and need to clear the global lists
    data_leaves.clear()
    tree_leaves.clear()
    print("Loading tree from " + str(filepath))
    with open(filepath) as f:
        data = json.load(f)

    # construct a tree from the json
    data_dict_root = data[0]
    tree_root = FungiNode(name=data_dict_root['id'], height=float(data_dict_root['x']))
    recursively_copy_children(data_dict_root, tree_root)

    print('tree constructed')
    return tree_root, tree_leaves


def recursively_copy_children(data_node, tree_node):
    if len(data_node['children']) > 0:
        # copy left child
        data_left = data_node['children'][0]
        tree_left = FungiNode(name=data_left['id'], parent=tree_node, height=float(data_left['x']))
        tree_node.left = tree_left
        recursively_copy_children(data_left, tree_left)
        if len(data_node['children']) > 1:
            # copy right child
            data_right = data_node['children'][1]
            tree_right = FungiNode(name=data_right['id'], parent=tree_node, height=float(data_right['x']))
            tree_node.right = tree_right
            recursively_copy_children(data_right, tree_right)
    else:  # indicates the current node is a leaf
        tree_node.is_leaf = True
        tree_node.name = data_node['name_col']  # this contains the actual species name
        data_leaves.append(data_node)
        tree_leaves.append(tree_node)


# this method both trims the tree and adds the lifestyle and feature annotations
def annotate_and_trim_tree(root, leaves, data_array):
    print('Pruning unused subtrees')
    # species = list(data_dict['Species'])
    # COLLECTING ONE SPECIES LABEL INSTEAD OF ALL
    species = [row[-2] for row in data_array]
    leaf_species = list()
    for node in leaves:
        leaf_species.append(node.name)
    intersection = set(species).intersection(set(leaf_species))
    if len(intersection) != len(species):
        # need to deal with these 4 examples at some point, seems like we could manually fix them
        print('Missing species:')
        for name in species:
            if name not in leaf_species:
                print(name)

    for leaf in leaves:
        if leaf.name not in species:
            leaf.is_present = False

    # first round of pruning
    for leaf in leaves:
        recursively_mark_for_prune(leaf.parent)

    recursively_prune(root)
    leaves = [leaf for leaf in leaves if leaf.is_present]

    # todo: do we need to add a bias feature?
    # now need to fill in the features
    for leaf in leaves:
        match = False
        for data_row in data_array:
            if leaf.name == data_row[-2]:  # the species name
                leaf.x = data_row[:-2]# the cluster counts
                leaf.y = data_row[-1]
                match = True
                break
        if not match:
            leaf.is_present = False

    leaves = [leaf for leaf in leaves if leaf.is_present]

    return root, leaves


# note that this cannot be called on leaves, it would mark all of them for not having children
def recursively_mark_for_prune(node):
    if (node.left is None or not node.left.is_present) and (node.right is None or not node.right.is_present):
        node.is_present = False
    if node.parent is not None:
        recursively_mark_for_prune(node.parent)


# Starts at root, works down. Assumes the root is present.
def recursively_prune(node):
    if node.left is not None:
        if not node.left.is_present:
            node.left = None
        else:
            recursively_prune(node.left)
    if node.right is not None:
        if not node.right.is_present:
            node.right = None
        else:
            recursively_prune(node.right)


def recursively_add_placement_vector(node, curr_depth, max_depth):
    if node.left is not None and node.right is not None:  # only making modifications on nodes with splits
        left_placement = copy.deepcopy(node.placement_vector)
        if curr_depth <= max_depth:
            left_placement.append(0.0)
        node.left.placement_vector = left_placement
        recursively_add_placement_vector(node.left, curr_depth=curr_depth+1, max_depth=max_depth)

        right_placement = copy.deepcopy(node.placement_vector)
        if curr_depth <= max_depth:
            right_placement.append(1.0)
        node.right.placement_vector = right_placement
        recursively_add_placement_vector(node.right, curr_depth=curr_depth+1, max_depth=max_depth)
    elif node.left is not None:
        left_placement = copy.deepcopy(node.placement_vector)
        node.left.placement_vector = left_placement
        recursively_add_placement_vector(node.left, curr_depth, max_depth)
    elif node.right is not None:
        right_placement = copy.deepcopy(node.placement_vector)
        node.right.placement_vector = right_placement
        recursively_add_placement_vector(node.right, curr_depth, max_depth)


def retrieve_feature_and_labels(target, features):
    """
    :param targets: binary target being trained on
    :param features: possibilities include smc, cazy, merops, transporters
    :return: preprocessed feature array with labels in the last column
    """
    data_array = list()
    feature_names = ['bias']
    parent = os.path.abspath('..')
    base_path = os.path.join(parent, 'fungi_application', 'data_files', 'Final_Database_v7')
    label_arr = pd.read_csv(os.path.join(base_path, 'LifestyleDatabase_v7_29Jul2020_PublishedAndRH-FMGenomes_SMCs_WithMellp2.csv'))
    label_col = label_arr[target]
    species_col = label_arr['Species']
    # note that this logic will EXCLUDE species with unknown label
    for lb, species in zip(label_col, species_col):
        if lb == 'YES':
            data_array.append([0.0, species, 1.0])  # 0.0 represents bias feature
        elif lb == 'NO':
            data_array.append([0.0, species, 0.0])
    feat_start = 54  # hard coding where all the feature start in v7 of the database
    for ft in reversed(features):
        if ft == 'smc':  # we have already opened it for the labels
            feature_names = list(label_arr.columns)[feat_start:] + feature_names
            feat_array = label_arr.to_numpy()
            add_feats(data_array, feat_array, feat_start)
        elif ft == 'cazy':
            feat_array = pd.read_csv(os.path.join(base_path, 'LifestyleDatabase_v7_29Jul2020_PublishedAndRH-FMGenomes_CAZys_WithMellp2.csv'))
            feature_names = list(feat_array.columns)[feat_start:] + feature_names
            add_feats(data_array, feat_array.to_numpy(), feat_start)
        elif ft == 'merops':
            feat_array = pd.read_csv(os.path.join(base_path, 'LifestyleDatabase_v7_29Jul2020_PublishedAndRH-FMGenomes_MEROPSFamiliesOnly_WithMellp2.csv'))
            feature_names = list(feat_array.columns)[feat_start:] + feature_names
            add_feats(data_array, feat_array.to_numpy(), feat_start)
        elif ft == 'transporters':
            feat_array = pd.read_csv(os.path.join(base_path, 'LifestyleDatabase_v7_29Jul2020_PublishedAndRH-FMGenomes_Transporters_WithMellp2.csv'))
            feature_names = list(feat_array.columns)[feat_start:] + feature_names
            add_feats(data_array, feat_array.to_numpy(), feat_start)
        elif ft == 'transfactors':
            feat_array = pd.read_csv(os.path.join(base_path, 'LifestyleDatabase_v7_29Jul2020_PublishedAndRH-FMGenomes_TransFactors_WithMellp2.csv'))
            feature_names = list(feat_array.columns)[feat_start:] + feature_names
            add_feats(data_array, feat_array.to_numpy(), feat_start)
        elif ft == 'plant_traits':  # we pull from the already opened smc frame, collect at appropriate columns
            feat_array = label_arr.to_numpy()
            feature_names = PLANT_TRAITS + feature_names
            feat_indices = [list(label_arr.columns).index(pt) for pt in PLANT_TRAITS]
            for p_idx, processed_row in zip(range(len(data_array)), data_array):
                for feat_row in feat_array:
                    if processed_row[-2] == feat_row[1]:  # checking if the species are a match
                        new_feats = [feat_row[i] for i in feat_indices]
                        for idx, nf in enumerate(new_feats):
                            if nf == 'YES':
                                new_feats[idx] = 1.0
                            else:  # assumes no missing positive samples
                                new_feats[idx] = 0.0
                        data_array[p_idx] = new_feats + processed_row

    return data_array, feature_names


def add_feats(data_array, feat_array, feat_start):
    for p_idx, processed_row in zip(range(len(data_array)), data_array):
        for feat_row in feat_array:
            if processed_row[-2] == feat_row[1]:  # checking if the species are a match
                new_feats = [float(i) for i in feat_row[feat_start:]]
                data_array[p_idx] = new_feats + processed_row


def parse_tree(target, feature_list, placement_depth=0):
    parent = os.path.abspath('..')
    tree_path = os.path.join(parent, 'fungi_application', 'data_files', 'phylotree.json')
    tree, leaves = load_tree(tree_path)  # August 2020: hard-coding in datapath for up to date dataset
    # create the feature matrix using tweaked function from layne-masters repo
    data_array, feature_names = retrieve_feature_and_labels(target, feature_list)
    # add features to tree leaves, trim where necessary
    trimmed_tree, trimmed_leaves = annotate_and_trim_tree(tree, leaves, data_array)

    recursively_add_placement_vector(trimmed_tree, curr_depth=0, max_depth=placement_depth)

    return trimmed_tree, trimmed_leaves, feature_names
