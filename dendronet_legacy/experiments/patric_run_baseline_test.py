"""
A classification experiment using the patric data
"""

import os, argparse
import pandas as pd
# import networkx as nx
import numpy as np
import tensorflow as tf
from collections import deque

# from networkx.drawing.nx_agraph import write_dot


from utils import generate_default_config, create_directory, dump_dict, plot_losses, log_aucs
from models.custom_models import MultifurcatingTreeModelLogReg, LogRegModel
from experiments.experiment_classes import LogRegExperiment
from patric_application.process_genome_lineage import create_labelled_tree, load_tree_and_leaves

# 2 15 17 22 36 42

# def print_tree_model(model, root, leaves, lifestyle=None):
#     g = nx.DiGraph()
#     id = 0  # pre-order BFS index, root starts at 0
#     q = deque()
#     # add the root to the graph, push it to the queue of nodes to process
#     name = str(str(id) + ": " + str(0.0))  # second number denotes the delta between this node and one above it
#     g.add_node(name)
#     q.append((model.root_layer, name))
#     id += 1
#     # full pre-order breadth first traversal
#     while len(q) > 0:
#         curr = q.popleft()
#         node = curr[0]
#         parent_id = curr[1]
#         # todo add the delta information in some way
#         if node.left is not None:
#             parent_weights = node.layer.trainable_weights[0].numpy()
#             weights = node.left.layer.trainable_weights[0].numpy()
#             delta = round(sum(np.abs(np.subtract(parent_weights, weights)).flatten()), 2)
#             name = str(str(id) + ": " + str(delta))
#             g.add_node(name)
#             g.add_edge(parent_id, name)
#             q.append((node.left, name))
#             id += 1
#         if node.right is not None:
#             parent_weights = node.layer.trainable_weights[0].numpy()
#             weights = node.right.layer.trainable_weights[0].numpy()
#             delta = round(sum(np.abs(np.subtract(parent_weights, weights)).flatten()), 2)
#             name = str(str(id) + ": " + str(delta))
#             g.add_node(name)
#             g.add_edge(parent_id, name)
#             q.append((node.right, name))
#             id += 1
#     name = 'tree.dot'
#     if lifestyle is not None:
#         name = lifestyle + '_' + name
#     write_dot(g, name)
#     """
#     Command to convert resulting file to a png using graphviz: "dot -Tpng tree.dot -o tree.png"
#     """
#
#     print('tree structure saved')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running patric experiment')
    parser.add_argument('--num-steps', type=int, default=10000, metavar='N',
                        help='number of steps for training (default: 1000)')
    parser.add_argument('--seed', type=int, default=[0], metavar='S',
                        help='random seed for train/valid split (default: 0)')
    parser.add_argument('--validation-interval', type=int, default=50, metavar='VI',
                        help='How often to run validation (default: 25 epochs)')
    parser.add_argument('--delta-penalty-factor', type=float, default=0.001, metavar='D',
                        help='scaling factor applied to delta term in the loss (default: 1.0)')
    parser.add_argument('--output-dir', type=str, default='test_gentamicin', metavar='O',
                        help='relative path to the directory for the output files (default: patric_single)')
    parser.add_argument('--antibiotic-index', type=int, default=0, metavar='I',
                        help='index of antibiotic of interest (default: 0)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--load-tree', type=bool, default=True, metavar='L',
                        help='Whether to load a tree from the specified folder or make a new one (default: True)')
    parser.add_argument('--tree-folder', type=str, default='patric_application/patric_tree_storage/gentamicin', metavar='T',
                        help='folder to look in for a stored tree structure')
    parser.add_argument('--label-file', type=str, default='gentamicin_firmicutes_samples.csv', metavar='LF',
                        help='file to look in for labels')
    parser.add_argument('--baselines', type=bool, default=True)
    args = parser.parse_args()

    config = generate_default_config()
    args_dict = vars(args)
    for key in args_dict.keys():
        if key is not 'seed' and key is not 'seeds':
            config[key] = args_dict[key]

    # directory removal warning!
    dir_name = str(args.output_dir)
    base_output_dir = os.path.join(os.path.abspath('.'), 'results', dir_name)
    create_directory(base_output_dir, remove_curr=True)

    labels_file = os.path.join(os.path.abspath('..'),
                               'patric_application/data_files', args.label_file)
    if args.load_tree:
        tree_folder = os.path.join(os.path.abspath('..'), args.tree_folder)
        data_root, leaves = load_tree_and_leaves(tree_folder)
    else:
        data_file = os.path.join(os.path.abspath('..'), 'patric_application/data_files/genome_lineage')
        data_root, leaves = create_labelled_tree(data_file, labels_file)

    # annotating leaves with labels and features
    labels_df = pd.read_csv(labels_file, dtype=str)
    match = False
    for row in labels_df.itertuples():
        for leaf in leaves:
            match = False
            if leaf.name == getattr(row, 'ID'):
                match = True
                if not (0 in eval(getattr(row, 'Phenotype'))):  # if any is resistant, label is resistant
                    leaf.y = 1.0
                else:
                    leaf.y = 0.0
                leaf.x = eval(getattr(row, 'Features'))
                """
                adding the other feature vectors for the new baseline methods
                """

                leaf.expanded_features = list(leaf.taxonomy_dict['safe_class'])
                leaf.expanded_features.extend(leaf.x)
                leaf.expanded_features = tf.identity(leaf.expanded_features)
                break
        if not match:
            print('missing features')

    expanded_x = list()
    for leaf in leaves:
        expanded_x.append(leaf.expanded_features)
        assert len(leaf.x) == len(leaves[0].x)
    expanded_x = tf.identity(expanded_x)
    weights_dim = len(leaves[0].x)
    output_dim = 2  # hard coded for binary classification
    layer_shape = (weights_dim, output_dim)
    if args.baselines:
        expanded_weights_dim = len(leaves[0].expanded_features)
        expanded_layer_shape = (expanded_weights_dim, output_dim)

    basenames = []
    if args.baselines:
        basenames.extend(['parsimony_best', 'parsimony_final', 'one_hot_best', 'one_hot_final'])
    aucs = dict()
    for name in basenames:
        aucs[name] = list()

    if isinstance(args_dict['seed'], int):
        args_dict['seed'] = [args_dict['seed']]

    for seed in args_dict['seed']:
        config['seed'] = seed
        output_dir = os.path.join(base_output_dir, 'seed' + str(seed))
        # directory removal warning!
        create_directory(output_dir, remove_curr=True)


        """
        Running the baselines model
        """
        if args.baselines:
            parsimony_model = MultifurcatingTreeModelLogReg(data_tree_root=data_root, leaves=leaves, layer_shape=(1, 2))
            one_hot_model = LogRegModel(layer_shape=expanded_layer_shape)
            one_hot_model.build((expanded_weights_dim, 1))

            experiment = LogRegExperiment(parsimony_model, one_hot_model, config, data_root, leaves,
                                                   baselines=args.baselines, expanded_x=expanded_x, use_test=True)
            """
            We are running on the test set, so we will combine the training and validation sets into a new training set,
            and use the test indices as the "validation set", reporting the final result at the end of training as our test 
            score
            """
            experiment.train_x = tf.concat((experiment.train_x, experiment.valid_x), axis=0)
            experiment.train_y = tf.concat((experiment.train_y, experiment.valid_y), axis=0)
            experiment.train_idx.extend(experiment.valid_idx)

            experiment.valid_x = experiment.test_x
            experiment.valid_y = experiment.test_y
            experiment.valid_idx = experiment.test_idx
            par_dendro_losses, par_prediction_losses, par_validation_losses, par_validation_aucs = experiment.train_dendronet()
            one_hot_prediction_losses, one_hot_validation_losses, one_hot_validation_aucs = experiment.train_simple_model()


        # tree_model = MultifurcatingTreeModelLogReg(data_tree_root=data_root, leaves=leaves, layer_shape=layer_shape)
        # simple_model = LogRegModel(layer_shape=layer_shape)
        # input_shape = (weights_dim, 1)
        # simple_model.build(input_shape)


        dump_dict(config, output_dir)
        # experiment = LogRegExperiment(tree_model, simple_model, config, data_root, leaves, use_test=True)
        # dendronet_losses, prediction_losses, validation_losses, validation_aucs = experiment.train_dendronet()
        # simple_prediction_losses, simple_validation_losses, simple_validation_aucs = experiment.train_simple_model()

        x_label = str(config['validation_interval']) + "s of steps"
        if args.baselines:
            plot_file = os.path.join(output_dir, 'parsimony_losses.png')
            plot_losses(plot_file, [par_dendro_losses, par_prediction_losses, par_validation_losses],
                        ['mutation', 'training', 'validation'], x_label)
            plot_file = os.path.join(output_dir, 'one_hot_losses.png')
            plot_losses(plot_file, [one_hot_prediction_losses, one_hot_validation_losses], ['training', 'validation'],
                        x_label)

        if len(par_validation_aucs) >= 1:
            if args.baselines:
                aucs['parsimony_best'].append(max(par_validation_aucs[1:]))
                aucs['one_hot_best'].append(max(one_hot_validation_aucs[1:]))
                aucs['parsimony_final'].append(par_validation_aucs[-1])
                aucs['one_hot_final'].append(one_hot_validation_aucs[-1])

    auc_list = list()
    log_auc_names = list()
    for name in basenames:
        auc_list.append(aucs[name])
        log_auc_names.append(name)
    if len(auc_list[0]) > 0:
        log_aucs(os.path.join(base_output_dir, 'auc_log.txt'), auc_list, log_auc_names)
    print('done')
