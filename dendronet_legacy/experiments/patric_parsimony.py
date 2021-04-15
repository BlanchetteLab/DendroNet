"""
A classification experiment using the patric data
"""

import os, argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import deque

from utils import generate_default_config, create_directory, dump_dict, plot_losses, log_aucs
from models.custom_models import MultifurcatingTreeModelLogReg
from experiments.experiment_classes import LogRegExperiment
from patric_application.process_genome_lineage import create_labelled_tree, load_tree_and_leaves





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running patric experiment')
    parser.add_argument('--num-steps', type=int, default=10000, metavar='N',
                        help='number of steps for training (default: 1000)')
    parser.add_argument('--seed', type=int, default=[0, 1, 2], metavar='S',
                        help='random seed for train/valid split (default: 0)')
    parser.add_argument('--validation-interval', type=int, default=50, metavar='VI',
                        help='How often to run validation (default: 25 epochs)')
    parser.add_argument('--delta-penalty-factor', type=float, default=0.05, metavar='D',
                        help='scaling factor applied to delta term in the loss (default: 1.0)')
    parser.add_argument('--use-l2', type=bool, default=False, help='Whether or not to apply an L2 penalty (Default: false)')
    parser.add_argument('--l2-penalty-factor', type=float, default=0.001, metavar='L2F',
                        help='scaling factor applied to l2 term in the loss (default: 0.0001)')
    parser.add_argument('--output-dir', type=str, default='clindamycin', metavar='O',
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
    expanded_x = list()
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

    basenames = []
    if args.baselines:
        basenames.extend(['parsimony_best', 'parsimony_final'])
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


            baseline_experiment = LogRegExperiment(parsimony_model, None, config, data_root, leaves,
                                                   baselines=args.baselines, expanded_x=None, use_test=True)
            par_dendro_losses, par_prediction_losses, par_validation_losses, par_validation_aucs = baseline_experiment.train_dendronet()


        dump_dict(config, output_dir)


        x_label = str(config['validation_interval']) + "s of steps"
        if args.baselines:
            plot_file = os.path.join(output_dir, 'parsimony_losses.png')
            plot_losses(plot_file, [par_dendro_losses, par_prediction_losses, par_validation_losses],
                        ['mutation', 'training', 'validation'], x_label)


        if len(par_validation_aucs) >= 1:
            if args.baselines:
                aucs['parsimony_best'].append(max(par_validation_aucs[1:]))
                aucs['parsimony_final'].append(par_validation_aucs[-1])


    auc_list = list()
    log_auc_names = list()
    for name in basenames:
        auc_list.append(aucs[name])
        log_auc_names.append(name)
    if len(auc_list[0]) > 0:
        log_aucs(os.path.join(base_output_dir, 'auc_log.txt'), auc_list, log_auc_names)
    print('done')
