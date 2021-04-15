import copy
import json
import numpy as np
import pandas as pd
from data_structures.gen_tree import FungiNode


"""
This file is for transforming the output of the script written by Alex and using it to create a binary phylogenetic
tree usable as input to the ML programs. Also need functionality for pruning out unneeded sub-trees
"""

# todo:  definitely refactor these out of being global variables
data_leaves = list()
tree_leaves = list()


def load_tree(filepath):
    # on the off chance we call this multiple times and need to clear the global lists
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
def annotate_and_trim_tree(root, leaves, lifestyle_data_path, feature_data_path, add_cluster_features=True):
    print('Pruning unused subtrees')
    data_dict = pd.read_csv(lifestyle_data_path)
    species = list(data_dict['Species'])
    lifestyles = list(data_dict['Lifestyle'])
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
        else:
            leaf.lifestyles = lifestyles[species.index(leaf.name)].split()

    # first round of pruning
    for leaf in leaves:
        recursively_mark_for_prune(leaf.parent)

    recursively_prune(root)
    leaves = [leaf for leaf in leaves if leaf.is_present]

    if add_cluster_features:
        # now need to fill in the features, transform lifestyle to target
        # clusters code taken from the fungi application preprocessor v2
        clusters_data = pd.read_csv(feature_data_path, delimiter="\t")
        clusters_data = clusters_data.fillna(0)
        cluster_headers = list(clusters_data.columns[0].split('     '))[1:]
        cluster_headers.append('bias_term')
        # cluster data need some more work, each row has been read in as a single string
        clusters_arr = np.asarray(clusters_data)
        cleaned_clusters_arr = list()
        for i in range(len(clusters_arr)):
            row = list(str(clusters_arr[i]).split('     '))
            row[0] = row[0][2:]  # formatting weirdness
            row[-1] = row[-1][:-2]
            for j in range(1, len(row)):
                if row[j] == '':
                    row[j] = 0
                else:
                    row[j] = int(row[j].strip())  # convert from strings to integers
                    """
                    From examining the data: there are 4 rows which have no cluster data, and the above logic results
                    in them having 13 0s as the feature data. We fix this by clipping off the incorrect extra zeros 
                    """
            cleaned_clusters_arr.append(row[0:9])
            cleaned_clusters_arr[-1].append(1.0)  # adding the bias term
        for leaf in leaves:
            match = False
            for cluster_row in cleaned_clusters_arr:
                if leaf.name == cluster_row[0]:  # the species name
                    leaf.x = [float(i) for i in cluster_row[1:]]  # the cluster counts
                    match = True
                    break
            if not match:
                leaf.is_present = False

        # second round of pruning for leaves without cluster feature data
        for leaf in leaves:
            recursively_mark_for_prune(leaf.parent)

        recursively_prune(root)
        leaves = [leaf for leaf in leaves if leaf.is_present]

        return root, leaves, cluster_headers

    else:
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


# the main method that is intended to be imported
def parse_tree(tree_path, data_path, feature_data_path, add_clusters=True, placement_depth=3):
    tree, leaves = load_tree(tree_path)
    if add_clusters:
        trimmed_tree, trimmed_leaves, feature_names = annotate_and_trim_tree(
            tree, leaves, data_path, feature_data_path, add_cluster_features=add_clusters)

    else:
        trimmed_tree, trimmed_leaves, feature_names = annotate_and_trim_tree(tree, leaves, data_path,
                                      feature_data_path=None, add_cluster_features=add_clusters)

    recursively_add_placement_vector(trimmed_tree, curr_depth=0, max_depth=placement_depth)

    return trimmed_tree, trimmed_leaves, feature_names
