"""
A regression experiment using the simulated entangled data
"""

import os, argparse
import itertools
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from utils import generate_default_config, create_directory, dump_dict, plot_losses, log_aucs
from models.custom_models import EntangledDeltaTreeModel, EntangledModel
from data_structures.entangled_data_simulation import BernoulliTree
from experiments.experiment_classes import RegressionExperiment
from scipy.stats import spearmanr


weights_dim = 16


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running entangled delta model experiment')
    parser.add_argument('--num-steps', type=int, default=100, metavar='N',
                        help='number of steps for training (default: 1000)')
    parser.add_argument('--validation-interval', type=int, default=50, metavar='VI',
                        help='How often to run validation (default: 25 epochs)')
    parser.add_argument('--seed', type=int, nargs='+', default=[11, 12, 13, 13, 14], metavar='S',
                        help='random seed for train/valid split (default: 0)')
    parser.add_argument('--depth', type=int, default=7, metavar='DE',
                        help='depth of the simulated tree (default: 8)')
    parser.add_argument('--num-leaves', type=int, default=1, metavar='L',
                        help='num samples to generate at each leaf of the tree (default: 1)')
    parser.add_argument('--delta-penalty-factor', type=float, default=0.75, metavar='D',
                        help='scaling factor applied to delta term in the loss (default: 1.0)')
    parser.add_argument('--use-l2', type=bool, default=False, help='Whether or not to apply an L2 penalty (Default: false)')
    parser.add_argument('--l2-penalty-factor', type=float, default=0.00001, metavar='L2F',
                        help='scaling factor applied to l2 term in the loss (default: 0.0001)')
    parser.add_argument('--output-dir', type=str, default='entangled_delta', metavar='O',
                        help='relative path to the directory for the output files (default: entangled_data)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--mr', type=float, default=0.4, metavar='MR',
                        help='mutation rate (default: 0.6)')
    parser.add_argument('--mp', type=float, default=1.0, metavar='MP',
                        help='mutation prob (default: 0.e)')
    parser.add_argument('--loss-scale', type=float, default=1.0, metavar='LS',
                        help='weight given to prediction loss (default: 1.0)')
    parser.add_argument('--ms', type=str, default='exponential',
                        help='type of mutation (normal or exponential)')
    parser.add_argument('--dls', type=str, default='l1',
                        help='type of mutation regularization (l1 or l1_indicator_approx)')
    parser.add_argument('--dcs1', type=float, default=0.0,
                        help='delta constant 1 (default: 0.0)')
    parser.add_argument('--dcs2', type=float, default=0.001,
                        help='delta constant 2 (default: 0.001)')
    args = parser.parse_args()

    config = generate_default_config()
    args_dict = vars(args)
    for key in args_dict.keys():
        if key is not 'seed' and key is not 'seeds':
            config[key] = args_dict[key]

    # directory removal warning!
    base_output_dir = os.path.join(os.path.abspath('.'), 'results', args.output_dir)
    create_directory(base_output_dir, remove_curr=True)

    # preprocessing steps
    # data_tree = BernoulliTree(mutation_rate=0.0, depth=args.depth, num_leaves=args.num_leaves, low=0.0, high=5.0,
    #                           mutation_prob=0.0, mutation_style='exponential')
    data_tree = BernoulliTree(mutation_rate=args.mr, depth=args.depth, num_leaves=args.num_leaves, low=0.0, high=5.0,
                              mutation_prob=args.mp, mutation_style=args.ms)
    data_root = data_tree.tree
    leaves = data_tree.leaves

    for leaf in leaves:
        leaf.x = list(leaf.x)
        leaf.x = np.asarray(leaf.x, dtype=np.float32)
    num_features = len(leaves[0].x)
    output_dim = 1  # hard coded for linear regression

    final_dendro_losses = list()
    best_dendro_corrs = list()
    best_dendro_errors = list()
    best_simple_corrs = list()
    best_simple_errors = list()
    best_dendro_norm_errors = list()
    best_simple_norm_errors = list()
    best_d_new_corrs = list()
    best_s_new_corrs = list()
    best_d_new_corrs_alt = list()
    best_abs_d_new_corrs = list()
    best_abs_d_new_corrs_alt = list()

    dendro_best_loss = list()
    dendro_final_loss = list()
    simple_best_loss = list()
    simple_final_loss = list()

    dendro_final_predictions = list()
    simple_final_predictions = list()
    targets = list()

    if isinstance(args_dict['seed'], int):
        args_dict['seed'] = [args_dict['seed']]
    for seed in args_dict['seed']:

        config['seed'] = seed
        output_dir = os.path.join(base_output_dir, 'seed' + str(seed))
        # directory removal warning!
        create_directory(output_dir, remove_curr=True)
        input_shape = (1, num_features)

        layer_shape = (num_features, output_dim)

        num_hidden_units = num_features
        simple_model = EntangledModel(output_dim, num_hidden_units, (num_features,))

        tree_model = EntangledDeltaTreeModel(data_root, leaves, weights_dim, delta_loss_style=args.dls,
                                             delta_const_1=args.dcs1, delta_const_2=args.dcs2)

        dump_dict(config, output_dir)
        experiment = RegressionExperiment(tree_model, simple_model, config, data_root, leaves)
        dendronet_losses, prediction_losses, validation_losses, _ = experiment.train_dendronet()
        simple_prediction_losses, simple_validation_losses, _ = experiment.train_simple_model()

        targets.extend(list(experiment.valid_y.numpy()))
        simple_final_predictions.extend([experiment.simple_model.call(x).numpy()[0][0] for x in experiment.valid_x])
        dendro_final_predictions.extend(list(tf.gather(experiment.tree_model.call(experiment.all_x)[0], experiment.valid_idx).numpy()))


        x_label = str(config['validation_interval']) + "s of steps"
        plot_file = os.path.join(output_dir, 'mutation_loss.png')
        plot_losses(plot_file, [dendronet_losses],
                    ['mutation'], x_label)
        plot_file = os.path.join(output_dir, 'dendronet_prediction_losses.png')
        plot_losses(plot_file, [prediction_losses, validation_losses],
                    ['training', 'validation'], x_label)
        plot_file = os.path.join(output_dir, 'simple_losses.png')
        plot_losses(plot_file, [simple_prediction_losses, simple_validation_losses], ['training', 'validation'], x_label)
        final_dendro_losses.append(dendronet_losses[-1] / config['delta_penalty_factor'])
        print('done experiment with seed ' + str(seed))

        """
        Retrieving weights and analysing disentanglement metrics for simple model
        For now sticking with the assumption that we have an diagonal matrix style generative process
        We do not know the configuration of factors->rows, so we charitably assume that the best possible
        configuration is the best one
        """
        simple_corrs = list()
        simple_errors = list()
        dendro_errors = list()
        dendro_corrs = list()
        dendro_norm_errors = list()
        simple_norm_errors = list()
        d_corrs_no_bias = list()
        d_new_corrs_alt = list()
        s_new_corrs = list()
        abs_d_corr_no_bias = list()
        abs_d_new_corrs_alt = list()

        configurations = list(itertools.permutations(range(num_features)))
        train_leaves = np.take(leaves, experiment.train_idx)
        for conf in configurations:
            # simple model normalized error


            # simple model, error and correlation
            weight_matrix = simple_model.trainable_variables[0].numpy()
            error = 0
            norm_error = 0
            feats = list()
            activations = list()
            n_activations = list()
            for i in range(num_features):
                error += sum(np.abs(weight_matrix[:, i])) - np.abs(weight_matrix[:, i][conf[i]]) # this is the entire col, minus the value of the causal position
            simple_errors.append(error)
            # adding analysis of normalized error
            for i in range(num_features):
                normalized_weight_vector =  [j / norm(weight_matrix[:, i]) for j in weight_matrix[:, i]]
                norm_error += sum(np.abs(normalized_weight_vector)) - np.abs(normalized_weight_vector[conf[i]]) # this is the entire row, minus the value of the causal position
            simple_norm_errors.append(norm_error)

            # note that one row on axis 0 of the weight_matrix represents weights for an output
            for leaf in train_leaves:
                for i in range(num_features):
                    feats.append(np.tanh(leaf.x[i] * leaf.a[i]))
                    n_activations.append(simple_model.layer_0.call(np.reshape(leaf.x, (1, 3))).numpy()[0][conf[i]])
                    activations.append(np.tanh(weight_matrix[:, i][conf[i]] * leaf.x[i]))
            simple_corrs.append(abs(spearmanr(feats, activations)[0]))
            s_new_corrs.append(abs(spearmanr(feats, n_activations)[0]))

            """
             dendro model, error and correlation
            """

            conf_errors = list()
            norm_conf_errors = list()
            d_feats = list()
            d_activations = list()
            d_n_activations = list()
            d_n_alt_activations = list()
            layers = list()
            biases = list()
            trained_activations = list()
            leaf_layers = tree_model.leaf_weights.numpy()
            leaf_biases = tree_model.leaf_biases.numpy()
            leaf_activations = tree_model.leaf_activations.numpy()
            for idx in experiment.train_idx:
                layers.append(leaf_layers[idx])
                biases.append(leaf_biases[idx])
                trained_activations.append(leaf_activations[idx])
            # layers = np.take(tree_model.leaf_weights.numpy(), experiment.train_idx)
            for leaf, layer, bias, trained_activation in zip(train_leaves, layers, biases, trained_activations):
                error = 0
                norm_error = 0
                weight_matrix = layer
                for i in range(num_features):
                    error += sum(np.abs(weight_matrix[:, i])) - np.abs(weight_matrix[:, i][conf[i]])  # this is the entire row, minus the value of the causal position
                conf_errors.append(error)
                for i in range(num_features):
                    normalized_weight_vector = [j / norm(weight_matrix[:, i]) for j in weight_matrix[:, i]]
                    norm_error += sum(np.abs(normalized_weight_vector)) - np.abs(normalized_weight_vector[conf[
                        i]])  # this is the entire row, minus the value of the causal position
                norm_conf_errors.append(norm_error)
                #
                for i in range(num_features):
                    d_feats.append(np.tanh(leaf.x[i] * leaf.a[i]))
                    d_activations.append(np.tanh(weight_matrix[:, i][conf[i]] * leaf.x[i]))
                    d_n_alt_activations.append(np.tanh(np.dot(weight_matrix[:, conf[i]], np.transpose(leaf.x)) + bias[conf[i]]))  # this one is confirmed correct

            dendro_corrs.append(abs(spearmanr(d_feats, d_activations)[0]))
            d_new_corrs_alt.append(abs(spearmanr(d_feats, d_n_alt_activations)[0]))
            abs_d_new_corrs_alt.append(abs(spearmanr(d_feats, np.abs(d_n_alt_activations))[0]))
            dendro_errors.append(np.mean(conf_errors))
            dendro_norm_errors.append(np.mean(norm_conf_errors))

        best_simple_errors.append(min(simple_errors))
        best_dendro_errors.append(min(dendro_errors))
        best_simple_corrs.append(max(simple_corrs))
        best_dendro_corrs.append(max(dendro_corrs))
        best_simple_norm_errors.append(min(simple_norm_errors))
        best_dendro_norm_errors.append(min(dendro_norm_errors))
        best_d_new_corrs_alt.append(max(d_new_corrs_alt))
        best_s_new_corrs.append(max(s_new_corrs))
        best_abs_d_new_corrs_alt.append(max(abs_d_new_corrs_alt))

        """
        We will also plot the best loss values similar to aucs for other experiments
        """
        if len(validation_losses) >= 1:
            dendro_best_loss.append(min(validation_losses[1:]))
            simple_best_loss.append(min(simple_validation_losses[1:]))
            dendro_final_loss.append(validation_losses[-1])
            simple_final_loss.append(simple_validation_losses[-1])

    if len(dendro_best_loss) > 0:
        names = ['dendro_best', 'dendro_final', 'simple_best', 'simple_final']
        loss_outputs = [dendro_best_loss, dendro_final_loss, simple_best_loss, simple_final_loss]
        log_aucs(os.path.join(base_output_dir, 'loss_log.txt'), loss_outputs, names)
    print('done')
    print('Done disentanglement analysis')

    disentanglement_metrics = {
        'simple_errors': best_simple_errors,
        'dendro_errors': best_dendro_errors,
        'simple_correlations': best_simple_corrs,
        'dendro_correlations': best_dendro_corrs,
        'dendro_norm_error': best_dendro_norm_errors,
        'simple_norm_errors': best_simple_norm_errors,
        'd_activation_corrs': best_d_new_corrs_alt,
        'abs_d_activation_corrs': best_abs_d_new_corrs_alt,
        's_activation_corr': best_s_new_corrs,
        'targets': targets,
        'final_dendro_predictions': dendro_final_predictions,
        'final_simple_predictions': simple_final_predictions
    }

    dump_dict(disentanglement_metrics, base_output_dir, name='disentanglement_metrics.csv')

    #constructing a dict for the dendronet loss analysis
    ideal_dendro_loss = data_tree.dendro_loss
    dendro_log = {'ideal_loss': ideal_dendro_loss,
                  'observed_mean': np.mean(final_dendro_losses),
                  'observed_std': np.std(final_dendro_losses)}
    for i in range(len(final_dendro_losses)):
        dendro_log['observation_' + str(i)] = final_dendro_losses[i]
    dump_dict(dendro_log, base_output_dir, name='dendro_log.csv')

