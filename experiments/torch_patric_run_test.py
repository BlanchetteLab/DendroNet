"""
A single-antibiotic classification experiment using the patric data and torch models
"""

import os
import copy
import argparse
import pandas as pd
import torch
from models.pytorch_models import LogRegModel, DendroLogReg
from experiments.torch_experiment_classes import ClassificationExperiment
from patric_application.process_genome_lineage import load_tree_and_leaves
from utils import generate_default_config, create_directory, dump_dict, plot_losses, log_aucs

# todo either pull in or adapt the print_tree_model function from tf implementation in patric_single.py

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running patric experiment')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of steps for training (default: 1000)')
    parser.add_argument('--seed', type=int, default=[0], metavar='S',
                        help='random seed for train/valid split (default: 0)')
    parser.add_argument('--validation-interval', type=int, default=1000, metavar='VI',
                        help='How often to run validation (default: 25 epochs)')
    parser.add_argument('--delta-penalty-factor', type=float, default=0.01, metavar='D',
                        help='scaling factor applied to delta term in the loss (default: 1.0)')
    parser.add_argument('--use-l2', type=bool, default=False, help='Whether or not to apply an L2 penalty (Default: false)')
    parser.add_argument('--run-baselines', type=bool, default=True)
    parser.add_argument('--l2-penalty-factor', type=float, default=0.01, metavar='L2F',
                        help='scaling factor applied to l2 term in the loss (default: 0.0001)')
    parser.add_argument('--output-dir', type=str, default='torch_test_penicillin', metavar='O',
                        help='relative path to the directory for the output files')
    parser.add_argument('--antibiotic-index', type=int, default=0, metavar='I',
                        help='index of antibiotic of interest (default: 0)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--tree-folder', type=str, default='patric_application/patric_tree_storage/penicillin', metavar='T',
                        help='folder to look in for a stored tree structure')
    parser.add_argument('--label-file', type=str, default='penicillin_firmicutes_samples.csv', metavar='LF',
                        help='file to look in for labels')
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

    tree_folder = os.path.join(os.path.abspath('..'), args.tree_folder)
    data_root, leaves = load_tree_and_leaves(tree_folder)

    # annotating leaves with labels and features
    labels_df = pd.read_csv(labels_file, dtype=str)
    match = False
    for row in labels_df.itertuples():
        for leaf in leaves:
            match = False
            if leaf.name == getattr(row, 'ID'):
                match = True
                if not (0 in eval(getattr(row, 'Phenotype'))):
                    leaf.y = 1.0
                else:
                    leaf.y = 0.0
                leaf.x = eval(getattr(row, 'Features'))
                leaf.x.append(1.0)  # adding a bias term
                """
                adding the other feature vectors for the new baseline methods
                """
                leaf.bias_feature = torch.tensor([1.0], dtype=torch.float32)
                # print('fix expanded features after tuning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                # leaf.expanded_features = torch.tensor(leaf.x, dtype=torch.float32)
                leaf.expanded_features = list(leaf.taxonomy_dict['safe_class'])
                leaf.expanded_features.extend(leaf.x)
                leaf.expanded_features = torch.tensor(leaf.expanded_features, dtype=torch.float32)
                break
        if not match:
            print('missing features')

    for leaf in leaves:
        assert len(leaf.x) == len(leaves[0].x)

    num_features = len(leaves[0].x)
    output_dim = 2  # hard coded for binary classification

    dendro_best_auc = list()
    dendro_final_auc = list()
    simple_best_auc = list()
    simple_final_auc = list()
    if args.run_baselines:
        phylog_best_auc = list()
        placement_best_auc = list()
        phylog_final_auc = list()
        placement_final_auc = list()

    if isinstance(args_dict['seed'], int):
        args_dict['seed'] = [args_dict['seed']]
    for seed in args_dict['seed']:
        config['seed'] = seed
        output_dir = os.path.join(base_output_dir, 'seed' + str(seed))
        # directory removal warning!
        create_directory(output_dir, remove_curr=True)

        tree_model = DendroLogReg(data_root=data_root, num_features=num_features)
        model_leaves = tree_model.leaf_list
        simple_model = LogRegModel(input_dim=num_features)

        dump_dict(config, output_dir)

        experiment = ClassificationExperiment(tree_model, simple_model, config, model_leaves, data_root, leaves,
                                              use_test=True)

        """
        We are running on the test set, so we will combine the training and validation sets into a new training set,
        and use the test indices as the "validation set", reporting the final result at the end of training as our test 
        score
        """

        experiment.train_x = torch.cat((experiment.train_x, experiment.valid_x), 0)
        experiment.train_y = torch.cat((experiment.train_y, experiment.valid_y), 0)
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

        if args.run_baselines:
            """
            Adding more baselines, eventually this could be consolidated into a single experiment object with some 
            refactoring of the train_dendronet and train_simple_model methods 
            """
            phylogeny_model = DendroLogReg(data_root=data_root, num_features=1)
            placement_model = LogRegModel(input_dim=len(leaves[0].expanded_features.detach().numpy()))
            baseline_experiment = ClassificationExperiment(phylogeny_model, placement_model, config, model_leaves,
                                                           data_root, leaves, use_test=True, baselines=True)
            phylogeny_dendro_losses, phylogeny_prediction_losses, phylogeny_validation_losses, phylogeny_validation_aucs = experiment.train_dendronet()
            placement_prediction_losses, placement_validation_losses, placement_validation_aucs = experiment.train_simple_model()

            """
            plotting additional baselines and storing baseline aucs
            """
            plot_file = os.path.join(output_dir, 'phylogeny.png')
            plot_losses(plot_file, [phylogeny_prediction_losses, phylogeny_validation_losses],
                        ['training', 'validation'], x_label)
            plot_file = os.path.join(output_dir, 'phylogeny_mutation_losses.png')
            plot_losses(plot_file, [phylogeny_dendro_losses],
                        ['mutation'], x_label)
            plot_file = os.path.join(output_dir, 'placement_losses.png')
            plot_losses(plot_file, [placement_prediction_losses, placement_validation_losses], ['training', 'validation'],
                        x_label)

        if len(validation_aucs) >= 1:
            dendro_best_auc.append(max(validation_aucs[1:]))
            simple_best_auc.append(max(simple_validation_aucs[1:]))
            dendro_final_auc.append(validation_aucs[-1])
            simple_final_auc.append(simple_validation_aucs[-1])

        if args.run_baselines:
            if len(phylogeny_validation_aucs) >= 1:
                phylog_best_auc.append(max(phylogeny_validation_aucs[1:]))
                placement_best_auc.append(max(placement_validation_aucs[1:]))
                phylog_final_auc.append(phylogeny_validation_aucs[-1])
                placement_final_auc.append(placement_validation_aucs[-1])

        # todo: port over tree visualization code (low priority)
        # print('visualizing tree!')
        # print_tree_model(tree_model, lifestyle=args.antibiotic, root=data_root, leaves=leaves)

    if len(dendro_best_auc) > 0:
        names = ['dendro_best', 'dendro_final', 'simple_best', 'simple_final']
        aucs = [dendro_best_auc, dendro_final_auc, simple_best_auc, simple_final_auc]
        if len(phylog_best_auc) > 0:
            names.extend(['phylog_best', 'phylog_final', 'placement_best', 'placement_final'])
            aucs.extend([phylog_best_auc, phylog_final_auc, placement_best_auc, placement_final_auc])
        log_aucs(os.path.join(base_output_dir, 'auc_log.txt'), aucs, names)

    print('done')