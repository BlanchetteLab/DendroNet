"""
A classification experiment using the patric data
"""

import os, argparse
import pandas as pd
import tensorflow as tf
from utils import generate_default_config, create_directory, dump_dict, plot_losses, log_aucs
from models.custom_models import MultifurcatingTreeModelLogReg, LogRegModel
from experiments.experiment_classes import LogRegExperiment
from patric_application.process_genome_lineage import create_labelled_tree, load_tree_and_leaves


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
    parser.add_argument('--use-l2', type=bool, default=False, help='Whether or not to apply an L2 penalty (Default: false)')
    parser.add_argument('--l2-penalty-factor', type=float, default=0.01, metavar='L2F',
                        help='scaling factor applied to l2 term in the loss (default: 0.0001)')
    parser.add_argument('--output-dir', type=str, default='clindamycin_old_test', metavar='O',
                        help='relative path to the directory for the output files (default: patric_single)')
    parser.add_argument('--antibiotic-index', type=int, default=0, metavar='I',
                        help='index of antibiotic of interest (default: 0)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--load-tree', type=bool, default=True, metavar='L',
                        help='Whether to load a tree from the specified folder or make a new one (default: True)')
    parser.add_argument('--tree-folder', type=str, default='patric_application/patric_tree_storage/clindamycin', metavar='T',
                        help='folder to look in for a stored tree structure')
    parser.add_argument('--label-file', type=str, default='clindamycin_firmicutes_samples.csv', metavar='LF',
                        help='file to look in for labels')
    args = parser.parse_args()

    config = generate_default_config()
    args_dict = vars(args)
    for key in args_dict.keys():
        if key is not 'seed' and key is not 'seeds':
            config[key] = args_dict[key]

    # directory removal warning!
    # dir_name = str(args.output_dir + str(args.antibiotic_index))
    dir_name = str(args.output_dir)
    base_output_dir = os.path.join(os.path.abspath('.'), 'results', dir_name)
    base_output_dir += '_test_set'
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
                break
        if not match:
            print('missing features')

    for leaf in leaves:
        assert len(leaf.x) == len(leaves[0].x)

    weights_dim = len(leaves[0].x)
    output_dim = 2  # hard coded for binary classification
    layer_shape = (weights_dim, output_dim)

    dendro_best_auc = list()
    dendro_final_auc = list()
    simple_best_auc = list()
    simple_final_auc = list()

    if isinstance(args_dict['seed'], int):
        args_dict['seed'] = [args_dict['seed']]
    for seed in args_dict['seed']:
        config['seed'] = seed
        output_dir = os.path.join(base_output_dir, 'seed' + str(seed))
        # directory removal warning!
        create_directory(output_dir, remove_curr=True)

        tree_model = MultifurcatingTreeModelLogReg(data_tree_root=data_root, leaves=leaves, layer_shape=layer_shape)
        simple_model = LogRegModel(layer_shape=layer_shape)
        # input_shape = (weights_dim, 1)
        # simple_model.build(input_shape)

        dump_dict(config, output_dir)
        experiment = LogRegExperiment(tree_model, simple_model, config, data_root, leaves, use_test=True)

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

        dendronet_losses, prediction_losses, validation_losses, validation_aucs = experiment.train_dendronet()
        simple_prediction_losses, simple_validation_losses, simple_validation_aucs = experiment.train_simple_model()

        x_label = str(config['validation_interval']) + "s of steps"
        plot_file = os.path.join(output_dir, 'dendronet_losses.png')
        plot_losses(plot_file, [prediction_losses, validation_losses],
                    ['training', 'validation'], x_label)
        plot_file = os.path.join(output_dir, 'mutation_losses.png')
        plot_losses(plot_file, [dendronet_losses],
                    ['mutation'], x_label)
        plot_file = os.path.join(output_dir, 'simple_losses.png')
        plot_losses(plot_file, [simple_prediction_losses, simple_validation_losses], ['training', 'validation'], x_label)

        if len(validation_aucs) >= 1:
            dendro_best_auc.append(max(validation_aucs[1:]))
            simple_best_auc.append(max(simple_validation_aucs[1:]))
            dendro_final_auc.append(validation_aucs[-1])
            simple_final_auc.append(simple_validation_aucs[-1])

    if len(dendro_best_auc) > 0:
        names = ['dendro_best', 'dendro_final', 'simple_best', 'simple_final']
        aucs = [dendro_best_auc, dendro_final_auc, simple_best_auc, simple_final_auc]
        log_aucs(os.path.join(base_output_dir, 'auc_log.txt'), aucs, names)
    print('done')
