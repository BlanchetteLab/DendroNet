import os
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from models.dendronet_models import DendroMatrixLogReg
from utils.utils import generate_default_config, create_directory
from utils.model_utils import build_parent_path_mat, split_indices
from fungi_application.preprocessing.parse_tree_json import parse_tree

"""
This file trains a DendroNet model on the JGI fungus tree, using various gene families as features sets 
(specified in the FEATURE_SETS list). Results stored compared for each of the target lifestyles as independent 
binary classification problems.
"""

# todo: add support for target trophic levels and feature sets as command line args
DIVERSE_TROPHIC_LEVELS = ['S', 'N', 'OB', 'HB', 'B', 'B_OB', 'Pathogenic', 'Plant']

FEATURE_SETS = [
    ['merops'],
    ['cazy'],
    ['smc'],
    ['transporters'],
    ['transfactors'],
    ['merops', 'cazy', 'smc'],
    ['merops', 'cazy', 'smc', 'transporters', 'transfactors']
]

USE_CUDA = True
print('Using CUDA: ' + str(USE_CUDA))
device = torch.device("cuda:0" if torch.cuda.is_available() and USE_CUDA else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running fungi experiment')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of training epochs (default: 1000)')
    parser.add_argument('--seed', type=int, default=[11, 12, 13, 14, 15, 16, 17, 18, 19], metavar='S',
                        help='random seed for train/valid split (default: 0)')
    parser.add_argument('--validation-interval', type=int, default=50, metavar='VI',
                        help='How often to run validation (default: 25 epochs)')
    parser.add_argument('--dpf', type=float, default=1.0, metavar='D',
                        help='scaling factor applied to delta term in the loss (default: 1.0)')
    parser.add_argument('--output-dir', type=str, default='fungi', metavar='O',
                        help='relative path to the directory for the output files (default: fungi)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    args = parser.parse_args()

    config = generate_default_config()
    args_dict = vars(args)
    for key in args_dict.keys():
        if key != 'seed' and key != 'seeds':
            config[key] = args_dict[key]

    for feature_set in FEATURE_SETS:

        # directory removal warning!
        base_output_dir = os.path.join(os.path.abspath('.'), 'results', args.output_dir, str(feature_set))
        create_directory(base_output_dir, remove_curr=True)

        # dicts for storing results across runs
        roc_metrics = dict()
        feature_importances = dict()

        roc_metric_names = ['dendro_roc']

        for ls in DIVERSE_TROPHIC_LEVELS:
            roc_metrics[ls] = dict()
            for name in roc_metric_names:
                roc_metrics[ls][name] = list()

        for ls in DIVERSE_TROPHIC_LEVELS:
            feature_importances[ls] = dict()
            feature_importances[ls]['feature_names'] = list()
            feature_importances[ls]['root_weights'] = list()
            feature_importances[ls]['effective_weights'] = list()

        if isinstance(args_dict['seed'], int):
            args_dict['seed'] = [args_dict['seed']]
        for seed in args_dict['seed']:
            config['seed'] = seed
            output_dir = os.path.join(base_output_dir, 'seed' + str(seed))
            # directory removal warning!
            create_directory(output_dir, remove_curr=True)

            for ls in DIVERSE_TROPHIC_LEVELS:
                data_root, _, feature_names = parse_tree(target=ls, feature_list=feature_set)
                """
                -using data_root (tree structure) to generate parent-child and parent-path matrices
                -as this is a non-balanced binary tree, we first count the amount of nodes present
                -then we fill parent-child in in-order traversal
                -then fill the parent-path matrix using the usual Dynamic Programming alg
                """

                # replaces leaves returned from parse_tree, we need to ensure that leaves
                # are identifiable by their col id in the pp mat
                leaves = list()

                node_queue = [data_root]
                node_list = list()
                while len(node_queue) > 0:
                    node_list.append(node_queue.pop(0))
                    if node_list[-1].is_leaf:
                        leaves.append((len(node_list)-1, node_list[-1]))
                    if node_list[-1].left is not None:
                        node_queue.append(node_list[-1].left)
                    if node_list[-1].right is not None:
                        node_queue.append(node_list[-1].right)
                num_nodes = len(node_list)
                num_edges = num_nodes - 1

                # constructing the parent-child matrix, would be nice to find a faster way to do this
                parent_child_mat = np.zeros(shape=(num_nodes, num_nodes), dtype=np.float32)
                for child_idx in range(1, len(node_list)):  # excluding the root
                    parent_idx = node_list.index(node_list[child_idx].parent)
                    parent_child_mat[parent_idx, child_idx] = 1.0

                pp_mat = build_parent_path_mat(parent_child_mat, num_edges=num_edges)

                # split the leaves into train and test
                train_idx, valid_idx = split_indices(range(len(leaves)))

                # constructing train and valid x and y matrices
                train_col_idx = [leaves[i][0] for i in train_idx]
                valid_col_idx = [leaves[i][0] for i in valid_idx]
                train_col_idx_tensor = torch.tensor(train_col_idx, device=device)
                valid_col_idx_tensor = torch.tensor(valid_col_idx, device=device)

                train_x = torch.tensor(np.asarray([leaves[i][1].x for i in train_idx]), device=device, dtype=torch.double)
                train_y = torch.tensor(np.asarray([leaves[i][1].y for i in train_idx]), device=device, dtype=torch.double)
                valid_x = torch.tensor(np.asarray([leaves[i][1].x for i in valid_idx]), device=device, dtype=torch.double)
                valid_y = torch.tensor(np.asarray([leaves[i][1].y for i in valid_idx]), device=device, dtype=torch.double)

                # filling spots missing data in the feature matrix
                train_x[torch.isnan(train_x)] = 0.0
                valid_x[torch.isnan(valid_x)] = 0.0

                num_features = len(feature_names)

                # tensor with the starting weights at the root node, assumes bias feature is included
                root_weights = np.zeros(num_features)

                # each column holds a trainable delta vector for each edge in the graph
                edge_tensor_matrix = np.zeros(shape=(num_features, num_edges))

                dendronet = DendroMatrixLogReg(device, root_weights, pp_mat, edge_tensor_matrix, init_deltas=False)

                # training loop
                loss_function = nn.BCELoss()
                if torch.cuda.is_available() and USE_CUDA:
                    loss_function = loss_function.cuda()
                optimizer = torch.optim.Adam(dendronet.parameters(), lr=args.lr)
                # training loop, could be optimized for batching
                for epoch in range(args.epochs):
                    # train pass
                    optimizer.zero_grad()
                    y_hat = dendronet.forward(train_x, train_col_idx_tensor)
                    delta_loss = dendronet.delta_loss()
                    train_loss = loss_function(y_hat, train_y)
                    loss = train_loss + (delta_loss * args.dpf)
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    if epoch % args.validation_interval == 0:
                        with torch.no_grad():
                            valid_y_hat = dendronet.forward(valid_x, valid_col_idx_tensor)
                            valid_loss = loss_function(valid_y_hat, valid_y)
                            print(epoch)
                            print('Delta loss', delta_loss * args.dpf)
                            print('Train loss', train_loss)
                            print('Train AUC', roc_auc_score(train_y.cpu(), y_hat.cpu().detach()))
                            print('Valid loss ', valid_loss)
                            print('Valid AUC', roc_auc_score(valid_y.cpu(), valid_y_hat.cpu().detach()))
                # capturing the final auc
                roc_metrics[ls]['dendro_roc'].append(roc_auc_score(valid_y.cpu(), valid_y_hat.cpu().detach()))
                if len(feature_importances[ls]['feature_names']) == 0:
                    feature_importances[ls]['feature_names'] = feature_names
                feature_importances[ls]['root_weights'].append([weight for weight in dendronet.root_weights.cpu().detach().numpy()])
                effective_weights = dendronet.get_effective_weights(list(range(num_nodes)))
                feature_importances[ls]['effective_weights'].append(effective_weights.cpu().detach().numpy().tolist())

        json_out = json.dumps(roc_metrics)
        open(os.path.join(base_output_dir, "auc_results.json"), "w").write(json_out)

        # dumping the feature names and feature weights at the root
        json_out = json.dumps(feature_importances)
        open(os.path.join(base_output_dir, "feature_importances.json"), "w").write(json_out)
