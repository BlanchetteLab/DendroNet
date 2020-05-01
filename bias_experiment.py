"""
This experiment trains a tree network using only a bias feature, essentially seeing how strong our results can get
from learning only the phylogenetic relationships enforced by the tree structure

In general this has a lot of replication from the fungi_experiment file, should be able to consolidate the two
eventually
"""


import os, argparse, shutil
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import train_valid_split_indices
from models.custom_models import TreeModelLogReg
from fungi_application.parse_tree_json import parse_tree

DIVERSE_TROPHIC_LEVELS = ['S', 'N', 'B', 'OB', 'HB', 'MP', 'P', 'ECM', 'E', 'L', 'C', 'T']
test_trophic_levels = ['S', 'N']


def annotate_leaves(leaf_list, target_lifestyle):
    for leaf in leaf_list:
        if target_lifestyle in leaf.lifestyles:
            leaf.target = 1.0
        else:
            leaf.target = 0.0
        leaf.features = [1.0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running logistic regression on the fungi data bias term')
    parser.add_argument('--num-steps', type=int, default=1000, metavar='N',
                        help='number of steps for training (default: 1000)')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed for train/valid split (default: None)')
    parser.add_argument('--delta-penalty-factor', type=float, default=1.0, metavar='D',
                        help='scaling factor applied to delta term in the loss (default: 1.0)')
    parser.add_argument('--output-dir', type=str, default='bias_experiment_output', metavar='O',
                        help='relative path to the directory for the output files (default: bias_experiment_output)')
    parser.add_argument('--log-file-name', type=str, default='auc_log_file', metavar='F',
                        help='name for file storing auc scores (default: auc_log_file.txt)')
    args = parser.parse_args()

    tree_path = os.path.join('fungi_application', 'data_files', 'phylotree.json')
    reference_path = os.path.join('fungi_application', 'data_files', 'All_clades_lifestyle_data_v2_Feb4_2019.csv')
    tree, leaves = parse_tree(tree_path, reference_path, feature_data_path=None, add_clusters=False)
    print('tree loaded!')
    if os.path.exists(args.output_dir):
        # todo: maybe find a less dangerous alternative
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    tree_aucs = list()
    for ls in DIVERSE_TROPHIC_LEVELS:
        print('Training with target ' + ls)
        annotate_leaves(leaf_list=leaves, target_lifestyle=ls)
        weights_dim = 1  # just the bias term
        output_dim = 2
        layer_shape = (weights_dim, output_dim)
        tree_model = TreeModelLogReg(data_tree_root=tree, leaves=leaves, layer_shape=layer_shape)

        """
        from here on there is a lot of repetition from the other experiment files, should seek to wrap 
        some of this in utility functions
        """
        train_indices, validation_indices = train_valid_split_indices(max_index=len(leaves), random_seed=args.seed)
        train_idx = tf.identity(train_indices)
        validation_idx = tf.identity(validation_indices)

        y_train = list()
        y_valid = list()
        leaf_x = list()
        for i in range(len(leaves)):
            leaf = leaves[i]
            leaf_x.append(leaf.features)
            if i in train_indices:
                y_train.append(leaf.target)
            else:
                y_valid.append(leaf.target)

        leaf_x = tf.identity(leaf_x)
        optimizer = tf.optimizers.Adam()
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        classification_loss = tf.keras.metrics.Mean(name='classification_loss')
        classification_accuracy = tf.metrics.Accuracy(name='classification_accuracy')
        regularization_loss = tf.keras.metrics.Mean(name='regularization_loss')
        validation_loss = tf.keras.metrics.Mean(name='validation_loss')
        validation_accuracy = tf.metrics.Accuracy(name='validation_accuracy')
        validation_auc = tf.metrics.AUC(name='validation_auc')

        @tf.function
        def train_step(x, y_t, y_v, calc_auc=True, penalty_factor=1.0):
            y_t = tf.keras.utils.to_categorical(y_t, num_classes=output_dim)
            y_v = tf.keras.utils.to_categorical(y_v, num_classes=output_dim)
            with tf.GradientTape() as tape:
                y_hat, delta = tree_model(x)
                # separate train examples from validation examples
                y_hat_train = tf.gather(y_hat, train_idx)
                y_hat_valid = tf.gather(y_hat, validation_idx)
                mutation_loss = penalty_factor* delta
                class_loss = tf.losses.categorical_crossentropy(y_t, y_hat_train)
                loss = class_loss + mutation_loss
                valid_loss = tf.losses.categorical_crossentropy(y_v,
                                                                y_hat_valid)  # note this does not get passed to optimizer
            gradients = tape.gradient(loss, tree_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, tree_model.trainable_variables))

            train_loss(tf.reduce_sum(loss))  # fudgy, sum of all losses
            classification_loss(tf.reduce_mean(class_loss))
            regularization_loss(tf.reduce_sum(mutation_loss))
            validation_loss(tf.reduce_mean(valid_loss))
            classification_accuracy.update_state(tf.argmax(y_t, 1), tf.argmax(y_hat_train, 1))
            validation_accuracy.update_state(tf.argmax(y_v, 1), tf.argmax(y_hat_valid, 1))
            if calc_auc:
                validation_auc.update_state(y_v[:,0:1], y_hat_valid[:,0:1])

        # store the losses for plotting
        mutation_losses = list()
        train_losses = list()
        validation_losses = list()

        for step in range(args.num_steps):
            # clear the states of the metrics that we want only single step measurements for
            validation_auc.reset_states()

            train_step(x=leaf_x, y_t=y_train, y_v=y_valid, penalty_factor=args.delta_penalty_factor)
            # if step == 1:
            # print(tree_model.summary())
            if step % 100 == 0:
                print('step ' + str(step))
                print('Loss: ' + str(train_loss.result().numpy()))
                mut_loss = regularization_loss.result().numpy()
                mutation_losses.append(mut_loss)
                print('Mutation loss: ' + str(mut_loss))
                trn_loss = classification_loss.result().numpy()
                train_losses.append(trn_loss)
                print('Train classification loss: ' + str(trn_loss))
                vld_loss = validation_loss.result().numpy()
                validation_losses.append(vld_loss)
                print('Validation classification loss: ' + str(vld_loss))
                class_accuracy = classification_accuracy.result().numpy()
                valid_accuracy = validation_accuracy.result().numpy()
                print('Train accuracy: ' + str(class_accuracy))
                print('Validation accuracy: ' + str(valid_accuracy))
                valid_auc = validation_auc.result().numpy()
                print('Validation AUC: ' + str(valid_auc))
                train_loss.reset_states()
                regularization_loss.reset_states()
                classification_loss.reset_states()
                validation_loss.reset_states()
                classification_accuracy.reset_states()
                validation_accuracy.reset_states()
                validation_auc.reset_states()

                print('')
                print('')
            if step + 1 == args.num_steps:
                tree_aucs.append(validation_auc.result().numpy())

        # plot and save results
        plt.plot(train_losses, label='train')
        plt.plot(validation_losses, label='validation')
        plt.xlabel('100\'s of steps')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.grid()
        name = ls + '_classification_loss.png'
        name = os.path.join(args.output_dir, name)
        plt.savefig(name)
        plt.clf()

        plt.plot(mutation_losses, label='mutation')
        plt.xlabel('100\'s of steps')
        plt.ylabel('loss')
        plt.grid()
        name = ls + '_mutation.png'
        name = os.path.join(args.output_dir, name)
        plt.savefig(name)
        plt.clf()

    #  writing out the file with the auc scores
    lines = list()
    for i in range(len(tree_aucs)):
        lines.append(str(DIVERSE_TROPHIC_LEVELS[i] + ':   ' + 'tree bias auc= ' + str(tree_aucs[i]) + '\n'))
    auc_file = open(os.path.join(args.output_dir, args.log_file_name), 'w+')
    auc_file.writelines(lines)
