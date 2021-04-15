"""
A classification experiment using the fungi data
"""
import copy
import pandas as pd
import os, argparse
import networkx as nx
import numpy as np
import tensorflow as tf
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from collections import deque
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from utils import generate_default_config, create_directory, dump_dict, plot_losses, log_aucs
from fungi_application.parse_tree_json import parse_tree
from models.custom_models import TreeModelLogReg, LogRegModel
from experiments.experiment_classes import LogRegExperiment

# DIVERSE_TROPHIC_LEVELS = ['S', 'N', 'B', 'OB', 'HB', 'MP', 'P', 'ECM', 'E', 'L', 'C', 'T']
DIVERSE_TROPHIC_LEVELS = ['S', 'N', 'OB', 'HB']
# DIVERSE_TROPHIC_LEVELS = ['OB', 'HB']
PLOT_ROC = True
FEATURE_INDEX = None


def get_tree_with_cluster_features(cluster_data_path, placement_depth=3):
    parent = os.path.abspath('..')
    tree_path = os.path.join(parent, 'fungi_application', 'data_files', 'phylotree.json')
    # reference_path = os.path.join(parent, 'fungi_application', 'data_files', 'All_clades_lifestyle_data_v2_Feb4_2019.csv')
    reference_path = os.path.join(parent, 'fungi_application', 'data_files', 'All_clades_lifestyle_data_v4_Jul12_2019_Genomes_referenced.csv')
    tree, leaves, feature_names = parse_tree(tree_path, reference_path, cluster_data_path, placement_depth=placement_depth)
    print('tree loaded!')
    return tree, leaves, feature_names


def annotate_with_targets(leaf_list, target_lifestyle):
    for leaf in leaf_list:
        if target_lifestyle in leaf.lifestyles:
            leaf.y = 1.0
        else:
            leaf.y = 0.0


def check_depth(node, depth=5):
    """
    messy but allows us to test the depth of a node without storing attributes
    """
    if node is not None:
        for _ in range(depth):
            if node.parent is not None:
                node = node.parent
            else:
                return False
        return True
    return False


def search_for_tree_starting_point(model_root, data_tree_root, feature_index=None, min_depth=5):
    curr_best_delta = 0.0
    curr_model_root = model_root.root_layer
    curr_data_root = data_tree_root

    model_q = deque()
    data_q = deque()
    model_q.append(curr_model_root)
    data_q.append(curr_data_root)
    while len(model_q) > 0:
        model_node = model_q.popleft()
        data_node = data_q.popleft()
        if check_depth(data_node, min_depth):
            dleft = 0
            dright = 0
            weights = model_node.layer.trainable_weights[0].numpy().flatten()
            if model_node.left is not None:
                child_weights = model_node.left.layer.trainable_weights[0].numpy().flatten()
                if feature_index is not None:
                    dleft = abs(weights[feature_index] - child_weights[feature_index])
                else:
                    dleft = abs(np.sum(np.subtract(weights, child_weights)))
            if model_node.right is not None:
                child_weights = model_node.right.layer.trainable_weights[0].numpy().flatten()
                if feature_index is not None:
                    dright = abs(weights[feature_index] - child_weights[feature_index])
                else:
                    dright = abs(np.sum(np.subtract(weights, child_weights)))
            if max(dleft, dright) > curr_best_delta:
                curr_best_delta = max(dleft, dright)
                curr_model_root = model_node
                curr_data_root = data_node

        if data_node.left is not None:
            model_q.append(model_node.left)
            data_q.append(data_node.left)
        if data_node.right is not None:
            model_q.append(model_node.right)
            data_q.append(data_node.right)

    return curr_model_root, curr_data_root


def print_tree_model(model_root, root, lifestyle=None, feature_index=None):
    g = nx.DiGraph()
    q = deque()
    data_q = deque()
    # add the root to the graph, push it to the queue of nodes to process
    data_node = root
    delta = 0.0
    # name = str(data_node.name+ "-" + str(delta))  # second number denotes the delta between this node and one above it
    name = str(data_node.name)  # second number denotes the delta between this node and one above it
    g.add_node(name)
    q.append((model_root, name))
    data_q.append((data_node, name))
    # full pre-order breadth first traversal
    while len(q) > 0:
        curr = q.popleft()
        node = curr[0]
        parent_id = curr[1]
        curr_data = data_q.popleft()
        data_node = curr_data[0]
        if node.left is not None:

            parent_weights = node.layer.trainable_weights[0].numpy()
            weights = node.left.layer.trainable_weights[0].numpy()
            if feature_index is None:
                delta = round(sum(np.abs(np.subtract(parent_weights, weights)).flatten()), 2)
            else:
                delta = round((parent_weights.flatten()[feature_index] - weights.flatten()[feature_index]), 2)
            # name = str(data_node.left.name + "-" + str(delta))
            name = str(data_node.left.name)
            if data_node.left.is_leaf:
                if data_node.left.y == 1.0:
                    g.add_node(name, color='red')
                else:
                    g.add_node(name, color='blue')
            else:
                g.add_node(name)

            g.add_edge(parent_id, name, weight=((1+abs(delta)) * 10))
            q.append((node.left, name))
            data_q.append((data_node.left, name))
        if node.right is not None:
            if data_node.right is None:
                print('error')
            parent_weights = node.layer.trainable_weights[0].numpy()
            weights = node.right.layer.trainable_weights[0].numpy()
            if feature_index is None:
                delta = round(sum(np.abs(np.subtract(parent_weights, weights)).flatten()), 2)
            else:
                delta = round((parent_weights.flatten()[feature_index] - weights.flatten()[feature_index]), 2)
            # name = str(data_node.right.name + "-" + str(delta))
            name = str(data_node.right.name)
            if data_node.right.is_leaf:
                if data_node.right.y == 1.0:
                    g.add_node(name, color='red')
                else:
                    g.add_node(name, color='blue')
            else:
                g.add_node(name)
            g.add_edge(parent_id, name, weight=((1+abs(delta)) * 10))
            q.append((node.right, name))
            data_q.append((data_node.right, name))
    name = 'tree.dot'
    # if lifestyle is not None:
        # if feature_index is not None:
        #     name = lifestyle + '_' + str(feature_index) + '_' + name
        # else:
        #     name = lifestyle + '_all_' + name
    name = 'full_tree_OB.dot'
    write_dot(g, name)

    plt.title('draw_networkx')
    pos = graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=False, arrows=True)
    plt.savefig('nx_test.png')
    """
    Command to convert resulting file to a png using graphviz: "dot -Tpng tree.dot -o tree.png"
    """
    print('tree structure saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running fungi experiment')
    parser.add_argument('--num-steps', type=int, default=1000, metavar='N',
    # parser.add_argument('--num-steps', type=int, default=60, metavar='N',
                        help='number of steps for training (default: 1000)')
    parser.add_argument('--seed', type=int, default=[11, 12, 13, 14, 15, 16, 17, 18, 19], metavar='S',
    # parser.add_argument('--seed', type=int, default=[11, 12], metavar='S',
                        help='random seed for train/valid split (default: 0)')
    parser.add_argument('--validation-interval', type=int, default=50, metavar='VI',
                        help='How often to run validation (default: 25 epochs)')
    parser.add_argument('--delta-penalty-factor', type=float, default=1.0, metavar='D',
                        help='scaling factor applied to delta term in the loss (default: 1.0)')
    parser.add_argument('--use-l2', type=bool, default=False, help='Whether or not to apply an L2 penalty (Default: false)')
    parser.add_argument('--baselines', type=bool, default=True)
    parser.add_argument('--l2-penalty-factor', type=float, default=0.001, metavar='L2F',
                        help='scaling factor applied to l2 term in the loss (default: 0.0001)')
    parser.add_argument('--output-dir', type=str, default='fungi_', metavar='O',
                        help='relative path to the directory for the output files (default: fungi)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lifestyle', type=str, default='OB')
    args = parser.parse_args()

    args.output_dir += args.lifestyle
    DIVERSE_TROPHIC_LEVELS = [args.lifestyle]

    config = generate_default_config()
    args_dict = vars(args)
    for key in args_dict.keys():
        if key is not 'seed' and key is not 'seeds':
            config[key] = args_dict[key]

    # directory removal warning!
    base_output_dir = os.path.join(os.path.abspath('.'), 'results', args.output_dir)
    create_directory(base_output_dir, remove_curr=True)

    cluster_data_path = os.path.join(os.path.abspath('..'), 'fungi_application', 'data_files', 'all_clusters.csv')
    data_root, leaves, feature_names = get_tree_with_cluster_features(cluster_data_path, placement_depth=1)  # increasing this causes vectors to be of varying length

    expanded_x = list()
    if args.baselines:
        for leaf in leaves:
            expanded_x.append(leaf.placement_vector)
            expanded_x[-1].extend(leaf.x)
        expanded_weights_dim = len(expanded_x[0])
        expanded_layer_shape = (expanded_weights_dim, 2)
        expanded_x = tf.identity(expanded_x)

    basenames = ['dendro_best', 'dendro_final', 'simple_best', 'simple_final']
    if args.baselines:
        basenames.extend(['parsimony_best', 'parsimony_final', 'one_hot_best', 'one_hot_final'])
    aucs = dict()
    for ls in DIVERSE_TROPHIC_LEVELS:
        aucs[ls] = dict()
        for name in basenames:
            aucs[ls][name] = list()

    roc_metric_names = ['targets', 'dendro_preds', 'simple_preds']
    if args.baselines:
        roc_metric_names.extend(['parsimony_preds', 'one_hot_preds'])
    roc_metrics = dict()
    for ls in DIVERSE_TROPHIC_LEVELS:
        roc_metrics[ls] = dict()
        for name in roc_metric_names:
            roc_metrics[ls][name] = list()

    if isinstance(args_dict['seed'], int):
        args_dict['seed'] = [args_dict['seed']]
    for seed in args_dict['seed']:
        config['seed'] = seed
        output_dir = os.path.join(base_output_dir, 'seed' + str(seed))
        # directory removal warning!
        create_directory(output_dir, remove_curr=True)

        for ls in DIVERSE_TROPHIC_LEVELS:
            print('Training with target ' + ls)
            annotate_with_targets(leaf_list=leaves, target_lifestyle=ls)
            positive_labels = 0
            for leaf in leaves:
                if leaf.y == 1.0:
                    positive_labels += 1

            print('Lifestyle ' + str(ls) + ' num positive examples: ' + str(positive_labels))

            weights_dim = len(feature_names)
            output_dim = 2
            layer_shape = (weights_dim, output_dim)

            """
            Running the baselines model
            """
            if args.baselines:
                parsimony_model = TreeModelLogReg(data_tree_root=data_root, leaves=leaves, layer_shape=(1, 2))
                one_hot_model = LogRegModel(layer_shape=expanded_layer_shape)
                one_hot_model.build((expanded_weights_dim, 1))
                baseline_config = copy.deepcopy(config)
                baseline_config['delta_penalty_factor'] = 0.25
                baseline_experiment = LogRegExperiment(parsimony_model, one_hot_model, baseline_config, data_root, leaves,
                                                       baselines=args.baselines, expanded_x=expanded_x)

                # """
                # temporary dumping preprocessed stuff for Yue
                # """
                # train_indices = baseline_experiment.train_idx
                # train_indices.extend(baseline_experiment.valid_idx)
                # train_matrix = tf.gather(expanded_x, train_indices).numpy()
                # train_y = np.concatenate((baseline_experiment.train_y, baseline_experiment.valid_y))
                # dump_dict = {
                #     'x_matrix': train_matrix,
                #     'labels': train_y,
                # }
                # name = 'fungi_preprocess_yue/' + ls
                # pd.DataFrame.from_records(train_matrix).to_csv(name + '_x_matrix.csv')
                # pd.DataFrame.from_records(train_y).to_csv(name + '_labels.csv')
                #
                # """
                # end temp dump
                # """

                par_dendro_losses, par_prediction_losses, par_validation_losses, par_validation_aucs = baseline_experiment.train_dendronet()
                one_hot_prediction_losses, one_hot_validation_losses, one_hot_validation_aucs = baseline_experiment.train_simple_model()

            tree_model = TreeModelLogReg(data_tree_root=data_root, leaves=leaves, layer_shape=layer_shape)
            simple_model = LogRegModel(layer_shape=layer_shape)
            input_shape = (weights_dim, 1)
            simple_model.build(input_shape)


            dump_dict(config, output_dir)
            experiment = LogRegExperiment(tree_model, simple_model, config, data_root, leaves)
            dendronet_losses, prediction_losses, validation_losses, validation_aucs = experiment.train_dendronet()
            simple_prediction_losses, simple_validation_losses, simple_validation_aucs = experiment.train_simple_model()

            """
            saving the tree graph
            """
            # tree_fig_model_root, tree_fig_data_root = search_for_tree_starting_point(tree_model, data_root, feature_index=FEATURE_INDEX)
            # print_tree_model(tree_fig_model_root, root=tree_fig_data_root, lifestyle=ls, feature_index=FEATURE_INDEX)
            tree_labels_file_name = 'OB_fig_labels.csv'
            dendro_predictions = tree_model.call(experiment.all_x)[0].numpy()
            simple_predictions = [simple_model.call(leaf.x).numpy() for leaf in leaves]
            fig_targets = [leaf.y for leaf in leaves]
            fig_species_names = [leaf.name for leaf in leaves]
            fig_dict = {
                'dendro_predictions': dendro_predictions,
                'simple_predictions': simple_predictions,
                'true_classifications': fig_targets,
                'species_names': fig_species_names
            }
            dump_dict(fig_dict, output_dir, name=tree_labels_file_name)

            # print_tree_model(tree_model.root_layer, root=data_root, lifestyle=ls, feature_index=FEATURE_INDEX)

            x_label = str(config['validation_interval']) + "s of steps"
            plot_file = os.path.join(output_dir, 'dendronet_losses.png')
            plot_losses(plot_file, [dendronet_losses, prediction_losses, validation_losses],
                        ['mutation', 'training', 'validation'], x_label)
            plot_file = os.path.join(output_dir, 'simple_losses.png')
            plot_losses(plot_file, [simple_prediction_losses, simple_validation_losses], ['training', 'validation'], x_label)
            if args.baselines:
                plot_file = os.path.join(output_dir, 'parsimony_losses.png')
                plot_losses(plot_file, [par_dendro_losses, par_prediction_losses, par_validation_losses],
                        ['mutation', 'training', 'validation'], x_label)
                plot_file = os.path.join(output_dir, 'one_hot_losses.png')
                plot_losses(plot_file, [one_hot_prediction_losses, one_hot_validation_losses], ['training', 'validation'], x_label)


            if len(simple_validation_aucs) >= 1:
                aucs[ls]['dendro_best'].append(max(validation_aucs[1:]))
                aucs[ls]['simple_best'].append(max(simple_validation_aucs[1:]))
                aucs[ls]['dendro_final'].append(validation_aucs[-1])
                aucs[ls]['simple_final'].append(simple_validation_aucs[-1])
                if args.baselines:
                    aucs[ls]['parsimony_best'].append(max(par_validation_aucs[1:]))
                    aucs[ls]['one_hot_best'].append(max(one_hot_validation_aucs[1:]))
                    aucs[ls]['parsimony_final'].append(par_validation_aucs[-1])
                    aucs[ls]['one_hot_final'].append(one_hot_validation_aucs[-1])
            """
            Collecting predictions from the trained model and plotting ROC curves
            """
            targets, dendro_predictions, simple_predictions = experiment.retrieve_predictions()
            roc_metrics[ls]['targets'].extend(targets)
            roc_metrics[ls]['dendro_preds'].extend(dendro_predictions.numpy()[:, 1])
            roc_metrics[ls]['simple_preds'].extend(simple_predictions.numpy()[:, 1])
            if args.baselines:
                _, parsimony_predictions, one_hot_predictions = baseline_experiment.retrieve_predictions()
                roc_metrics[ls]['parsimony_preds'].extend(np.squeeze(parsimony_predictions.numpy())[:, 1])
                roc_metrics[ls]['one_hot_preds'].extend(one_hot_predictions.numpy()[:, 1])

    auc_list = list()
    log_auc_names = list()
    for ls in DIVERSE_TROPHIC_LEVELS:
        for name in basenames:
            auc_list.append(aucs[ls][name])
            log_auc_names.append(str(ls + '_' + name))
    if len(auc_list[0]) > 0:
        log_aucs(os.path.join(base_output_dir, 'auc_log.txt'), auc_list, log_auc_names)
    print('done')

    if PLOT_ROC:
        for ls in DIVERSE_TROPHIC_LEVELS:
            pd_name = ls +'_roc_metrics.csv'
            pd.DataFrame.from_dict(roc_metrics[ls]).to_csv(pd_name)
            targets = roc_metrics[ls]['targets']
            dendro_predictions = roc_metrics[ls]['dendro_preds']
            simple_predictions = roc_metrics[ls]['simple_preds']
            bin_targets = [t[1] for t in targets]

            dendro_fpr, dendro_tpr, _ = roc_curve(bin_targets, dendro_predictions)
            simple_fpr, simple_tpr, _ = roc_curve(bin_targets, simple_predictions)

            dendro_auc = auc(dendro_fpr, dendro_tpr)
            simple_auc = auc(simple_fpr, simple_tpr)

            if args.baselines:
                par_predictions = roc_metrics[ls]['parsimony_preds']
                one_hot_predictions = roc_metrics[ls]['one_hot_preds']
                par_fpr, par_tpr, _ = roc_curve(bin_targets, par_predictions)
                one_hot_fpr, one_hot_tpr, _ = roc_curve(bin_targets, one_hot_predictions)
                par_auc = auc(par_fpr, par_tpr)
                one_hot_auc = auc(one_hot_fpr, one_hot_tpr)

            plt.title('ROC Curves for Target Class ' + ls)
            plt.plot(dendro_fpr, dendro_tpr, label="Dendronet, auc=" + str(round(dendro_auc, 2)), color='black', linestyle='solid')
            plt.plot(simple_fpr, simple_tpr, label="LogReg, auc=" + str(round(simple_auc, 2)), color='black', linestyle='dashed')
            if args.baselines:
                plt.plot(par_fpr, par_tpr, label='Parsimony, auc=' + str(round(par_auc, 2)), color='black', linestyle='dashdot')
                plt.plot(one_hot_fpr, one_hot_tpr, label='Placement, auc=' + str(round(one_hot_auc, 2)), color='black', linestyle='dotted')
            plt.legend(loc='best')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.savefig('roc_' + ls + '.png')
            plt.clf()

