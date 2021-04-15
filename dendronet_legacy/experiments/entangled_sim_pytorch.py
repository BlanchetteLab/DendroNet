"""
This is the pytorch version of the "entangled_delta_model.py" file implemented in tensorflow
Expected to be ~10 times faster
"""
import os, argparse, itertools
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
from experiments.torch_experiment_classes import RegressionExperiment
from models.pytorch_models import DendroFCNN, SimpleFCNN
from data_structures.entangled_data_simulation import BernoulliTree
from utils import generate_default_config, create_directory, dump_dict, plot_losses, log_aucs

weights_dim = 16
USE_CUDA=False

# [0.0, 0.25, 0.5, 1.0, 2.0]
# [0.0, 0.1, 1.0, 1.0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running entangled delta model experiment')
    parser.add_argument('--epochs', type=int, default=6000, metavar='N')
    parser.add_argument('--validation-interval', type=int, default=25, metavar='VI',
                        help='How often to run validation (default: 25 epochs)')
    parser.add_argument('--seed', type=int, nargs='+', default=[0, 1, 2, 4, 5, 6], metavar='S',
                        help='random seed for train/valid split (default: 0)')
    parser.add_argument('--depth', type=int, default=4, metavar='DE',
                        help='depth of the simulated tree (default: 8)')
    parser.add_argument('--num-leaves', type=int, default=16, metavar='L',
                        help='num samples to generate at each leaf of the tree (default: 2)')
    parser.add_argument('--delta-penalty-factor', type=float, default=0.1, metavar='D',
                        help='scaling factor applied to delta term in the loss (default: 1.0)')
    parser.add_argument('--output-dir', type=str, default='torch_entangled_delta', metavar='O',
                        help='relative path to the directory for the output files (default: torch_entangled_delta)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--mr', type=float, default=1.5, metavar='MR',
                        help='mutation rate (default: 0.6)')
    parser.add_argument('--mp', type=float, default=1.0, metavar='MP',
                        help='mutation prob (default: 0.e)')
    parser.add_argument('--ms', type=str, default='exponential',
                        help='type of mutation (normal or exponential)')
    parser.add_argument('--dls', type=str, default='l1',
                        help='type of mutation regularization (l1 or l1_indicator_approx)')
    parser.add_argument('--ind_leaves', type=bool, default=True)
    args = parser.parse_args()

    device=torch.device('cpu')
    if USE_CUDA:
        if torch.cuda.is_available():
            print('Found GPU device')
        else:
            print('Failed to find available GPU!!!')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = generate_default_config()
    args_dict = vars(args)
    for key in args_dict.keys():
        if key is not 'seed' and key is not 'seeds':
            config[key] = args_dict[key]

    # directory removal warning!
    base_output_dir = os.path.join(os.path.abspath('.'), 'results', args.output_dir)
    create_directory(base_output_dir, remove_curr=True)

    """
    April 14: with the new independent leaves, we are going to try a new range for starting A values. 0.0 anf 5.0 were
    used for all results in the Jan 30th submission
    """
    # data_tree = BernoulliTree(mutation_rate=args.mr, depth=args.depth, num_leaves=args.num_leaves, low=0.0, high=5.0,
    #                           mutation_prob=args.mp, mutation_style=args.ms, ind_leaves=args.ind_leaves)

    data_tree = BernoulliTree(mutation_rate=args.mr, depth=args.depth, num_leaves=args.num_leaves, low=-1.0, high=1.0,
                              mutation_prob=args.mp, mutation_style=args.ms, ind_leaves=args.ind_leaves)
    data_root = data_tree.tree
    leaves = data_tree.leaves
    print(str(len(leaves)) + ' leaves in the tree')

    num_features = len(leaves[0].x)

    final_dendro_losses = list()
    best_dendro_corrs = list()
    best_dendro_errors = list()
    best_simple_corrs = list()
    best_simple_errors = list()

    dendro_best_loss = list()
    dendro_final_loss = list()
    simple_best_loss = list()
    simple_final_loss = list()

    # simple_final_weights = list()
    # dendro_final_weights = list()
    # leaf_a_values = list()
    # leaf_y_values = list()
    # diagnostic_dict = dict()
    # todo: may need these later
    # dendro_final_predictions = list()
    # simple_final_predictions = list()
    # targets = list()

    if isinstance(args_dict['seed'], int):
        args_dict['seed'] = [args_dict['seed']]
    for seed in args_dict['seed']:
        config['seed'] = seed
        output_dir = os.path.join(base_output_dir, 'seed' + str(seed))
        # directory removal warning!
        create_directory(output_dir, remove_curr=True)

        tree_model = DendroFCNN(data_root=data_root, use_cuda=USE_CUDA, device=device)
        model_leaves = tree_model.leaf_list

        simple_model = SimpleFCNN(input_dim=num_features)
        if USE_CUDA:
            tree_model.to(device)
            simple_model.to(device)

        dump_dict(config, output_dir)
        experiment = RegressionExperiment(tree_model, simple_model, config, model_leaves, data_root, leaves,
                                          use_test=False)
        # train_leaf_starting_weights = list()
        # for i in range(3):
        #     train_leaf_starting_weights.append(model_leaves[2].weight_list[i].detach().numpy())
        delta_losses, prediction_losses, validation_losses, _ = experiment.train_dendronet()
        simple_prediction_losses, simple_validation_losses, _ = experiment.train_simple_model()
        # todo: omitted a block of code that collects that targets and final predictions for every model at every seed
        train_leaves = np.take(leaves, experiment.train_idx)
        train_model_leaves = np.take(model_leaves, experiment.train_idx)

        # plotting losses
        x_label = str(config['validation_interval']) + "s of steps"
        plot_file = os.path.join(output_dir, 'delta_loss.png')
        plot_losses(plot_file, [delta_losses],
                    ['delta_loss'], x_label)
        plot_file = os.path.join(output_dir, 'dendronet_prediction_losses.png')
        plot_losses(plot_file, [prediction_losses, validation_losses],
                    ['training', 'validation'], x_label)
        plot_file = os.path.join(output_dir, 'simple_losses.png')
        plot_losses(plot_file, [simple_prediction_losses, simple_validation_losses], ['training', 'validation'],
                    x_label)
        plot_file = os.path.join(output_dir, 'validation_loss_comparison.png')
        plot_losses(plot_file, [validation_losses, simple_validation_losses], ['dendro_valid', 'baseline_valid'],
                    x_label)
        # todo: omitted analysis of final delta loss vs ideal

        """
        Retrieving weights and analysing disentanglement metrics for simple model
        For now sticking with the assumption that we have an diagonal matrix style generative process
        We do not know the configuration of factors->rows, so we charitably assume that the best possible
        configuration is the best one
        """

        # retrieving weight vector for a dendronet leaf:
        # leaf_weights_dict = tree_model.return_weights(model_leaves[0])
        lin_weight_dict = {
            'fc1_w': simple_model.lin_1.weight,
            'fc1_b': simple_model.lin_1.bias,
            'fc2_w': simple_model.out_layer.weight,
            'fc2_b': simple_model.out_layer.bias
        }

        # """
        # collecting a sample of the final weights at different leaves
        # """
        # diagnostic_dict['mr'] = args.mr
        # diagnostic_dict['dpf'] = args.delta_penalty_factor
        # diagnostic_dict['simple_final_weights'] = lin_weight_dict
        # leaf_a_vals = list()
        # leaf_y_vals = list()
        # dendro_preds = list()
        # simple_preds = list()
        # # simple_final_weights.append(lin_weight_dict)
        # for i in range(min(10, len(train_model_leaves))):
        #     dendro_final_weights.append(tree_model.return_weights(train_model_leaves[i]))
        #     leaf_a_vals.append(train_leaves[i].a[0:3])
        #     leaf_y_vals.append(train_leaves[i].y)
        #     dendro_preds.append(float(tree_model.forward(train_model_leaves[i], torch.tensor(train_leaves[i].x, dtype=torch.float32)).detach().numpy()))
        #     simple_preds.append(simple_model.forward(torch.tensor(train_leaves[i].x, dtype=torch.float32)).detach().numpy()[0])
        # diagnostic_dict['dendro_final_weights'] = dendro_final_weights
        # diagnostic_dict['leaf_a_values'] = leaf_a_vals
        # diagnostic_dict['leaf_y_values'] = leaf_y_vals
        # diagnostic_dict['dendro_predictions'] = dendro_preds
        # diagnostic_dict['simple_predictions'] = simple_preds


        configurations = list(itertools.permutations(range(num_features)))

        """
        we are taking the equivalent of abs_d_activation_corrs and s_activation_corr from entangled_delta_model.py file
        """
        simple_corrs = list()
        simple_errors = list()
        dendro_corrs = list()
        dendro_errors = list()

        with torch.no_grad():
            """
            collecting some of the raw weights for analysis
            
            """
            for conf in configurations:
                """
                simple / baseline model error and activation correlations across all leaves
                """
                error = 0
                feats = list()
                activations = list()
                """
                calculating the 'sparse error' in the weights at non-causal positions
                """
                for i in range(num_features):
                    error += sum(np.abs(lin_weight_dict['fc1_w'][:, i].detach().numpy())) - np.abs(
                        lin_weight_dict['fc1_w'][:, i].detach().numpy()[conf[i]])  # this is the entire col, minus the value of the causal position
                simple_errors.append(error)
                """
                collecting features from data generation and equivalent activations
                """
                for leaf in train_leaves:
                    for i in range(num_features):
                        feats.append(np.tanh(leaf.x[i] * leaf.a[i]))
                        activations.append(simple_model.lin_1(torch.tensor(leaf.x, dtype=torch.float32)).detach().numpy()[conf[i]])
                simple_corrs.append(abs(spearmanr(feats, activations)[0]))

                """
                dendronet sparse errors and correlation metrics
                """
                conf_errors = list()
                d_feats = list()
                d_activations = list()
                for data_leaf, model_leaf in zip(train_leaves, train_model_leaves):
                    leaf_weight_dict = tree_model.return_weights(model_leaf)
                    # collecting sparse error for the leaf
                    error = 0
                    for i in range(num_features):
                        error += sum(np.abs(leaf_weight_dict['fc1_w'][:, i])) - \
                                 np.abs(leaf_weight_dict['fc1_w'][:, i][conf[i]])  # row minus causal position
                    conf_errors.append(error)

                    # collecting features and activations for the leaf
                    for i in range(num_features):
                        d_feats.append(np.tanh(data_leaf.x[i] * data_leaf.a[i]))
                        d_activations.append(nn.functional.linear(torch.tensor(data_leaf.x, dtype=torch.float32),
                                                                  weight=torch.tensor(leaf_weight_dict['fc1_w']),
                                                                  bias=torch.tensor(leaf_weight_dict['fc1_b']))[conf[i]])
                dendro_errors.append(np.mean(conf_errors))
                dendro_corrs.append(abs(spearmanr(d_feats, d_activations)[0]))

            best_simple_errors.append(min(simple_errors))
            best_dendro_errors.append(min(dendro_errors))
            best_simple_corrs.append(max(simple_corrs))
            best_dendro_corrs.append(max(dendro_corrs))

            """
            Plotting best & final validation losses
            """
            if len(validation_losses) > 0:
                dendro_best_loss.append(min(validation_losses[1:]))
                simple_best_loss.append(min(simple_validation_losses[1:]))
                dendro_final_loss.append(validation_losses[-1])
                simple_final_loss.append(simple_validation_losses[-1])
    if len(dendro_best_loss) > 0:
        names = ['dendro_best', 'dendro_final', 'simple_best', 'simple_final']
        loss_outputs = [dendro_best_loss, dendro_final_loss, simple_best_loss, simple_final_loss]
        log_aucs(os.path.join(base_output_dir, 'loss_log.txt'), loss_outputs, names)
    print('Done disentanglement analysis')

    disentanglement_metrics = {
        'simple_errors': best_simple_errors,
        'dendro_errors': best_dendro_errors,
        'simple_correlations': best_simple_corrs,
        'dendro_correlations': best_dendro_corrs,
        # 'simple_final_weights': simple_final_weights,
        # 'dendro_final_weights': dendro_final_weights
    }

    dump_dict(disentanglement_metrics, base_output_dir, name='disentanglement_metrics.csv')

    # dump_dict(diagnostic_dict, base_output_dir, name='diagnostic_dict.csv')