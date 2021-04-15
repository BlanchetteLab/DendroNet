"""
utilities functions, mostly for data wrangling or plotting
"""

import os
import csv
import math
import random
import string
import shutil
import torch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def train_valid_split_indices(max_index, validation_percentage=0.3, min_index=0, random_seed=None):
    """
    Given a range of data examples, randomly assigns indices to being either for training or validation
    :param max_index: max index to be assigned (usually length of the data array)
    :param validation_percentage: probability that any individual index will be assigned to the validation set
    :param min_index: starting index, defaults to 0
    :return: list of train indices, list of validation indices
    """
    assert(max_index - min_index) > 1, 'Too few examples'
    print('Using random seed ' + str(random_seed) + ' for train/valid split')
    np.random.seed(seed=random_seed)
    train_indices = list()
    valid_indices = list()
    for i in range(min_index, max_index):
        if np.random.uniform() < validation_percentage:
            valid_indices.append(i)
        else:
            train_indices.append(i)
    # sanity check that both lists have at least one index, prevents error with small test datasets
    if len(train_indices) == 0:
        train_indices.append(valid_indices.pop())
    elif len(valid_indices) == 0:
        valid_indices.append(train_indices.pop())
    return train_indices, valid_indices


def recover_model_weights(layer_node, weight_array):
    for i in range(len(layer_node.layer.weights)):
        weight_array[i].append(layer_node.layer.weights[i].numpy())

    for child in layer_node.descendants:
        recover_model_weights(child, weight_array)


def dataset_split_with_test(leaves, seed, test_seed=1, test_percentage=0.25, validation_percentage=0.35,
                            torch_format=False):
    """
    :param leaves:
    :param seed: random seed for splitting non-test set into train and validation sets
    :param test_seed: seed for splitting all samples into test and non-test set
    :param test_percentage:
    :param validation_percentage: Note that this is a percentage of the non-test examples, not the total samples
    :return:
    """
    non_test_indices, test_indices = train_valid_split_indices(max_index=len(leaves), random_seed=test_seed,
                                                               validation_percentage=test_percentage)

    # split the non-test set into train and validation, extract from non_test_indices list
    train_indices, validation_indices = train_valid_split_indices(max_index=len(non_test_indices), random_seed=seed,
                                                                  validation_percentage=validation_percentage)
    train_indices = [non_test_indices[i] for i in train_indices]
    validation_indices = [non_test_indices[i] for i in validation_indices]

    y_train = list()
    y_valid = list()
    y_test = list()
    x_train = list()
    x_valid = list()
    x_test = list()
    all_x = list()
    for i in range(len(leaves)):
        leaf = leaves[i]
        all_x.append(leaf.x)
        if i in train_indices:
            y_train.append(leaf.y)
            x_train.append(leaf.x)
        elif i in validation_indices:
            y_valid.append(leaf.y)
            x_valid.append(leaf.x)
        else:
            y_test.append(leaf.y)
            x_test.append(leaf.x)

    if torch_format:
        return torch.tensor(all_x, dtype=torch.float32), torch.tensor(x_train, dtype=torch.float32), \
               torch.tensor(x_valid, dtype=torch.float32), torch.tensor(x_test, dtype=torch.float32), \
               torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32), \
               torch.tensor(y_test, dtype=torch.float32), train_indices, validation_indices, test_indices

    return tf.identity(all_x), tf.identity(x_train), tf.identity(x_valid), tf.identity(x_test), tf.identity(y_train), \
        tf.identity(y_valid), tf.identity(y_test), train_indices, validation_indices, test_indices


def dataset_split(leaves, seed, torch_format=False):
    train_indices, validation_indices = train_valid_split_indices(max_index=len(leaves),
                                                                  random_seed=seed)
    y_train = list()
    y_valid = list()
    x_train = list()
    x_valid = list()
    all_x = list()
    for i in range(len(leaves)):
        leaf = leaves[i]
        all_x.append(leaf.x)
        if i in train_indices:
            y_train.append(leaf.y)
            x_train.append(leaf.x)
        else:
            y_valid.append(leaf.y)
            x_valid.append(leaf.x)

    if torch_format:
        return torch.tensor(all_x, dtype=torch.float32), torch.tensor(x_train, dtype=torch.float32), \
               torch.tensor(x_valid, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), \
               torch.tensor(y_valid, dtype=torch.float32), train_indices, validation_indices

    return tf.identity(all_x), tf.identity(x_train), tf.identity(x_valid), tf.identity(y_train), \
        tf.identity(y_valid), train_indices, validation_indices


def generate_default_config():
    return {
        'num_classes': 2,
        'lr': 0.001,
        'seed': 0,
        'l2': False,
        'delta_penalty_factor': 0.01,
        'l2_penalty_factor': 0.01,
        'num_steps': 5000,
        'validation_interval': 100,
        'loss_scale': 1.0
    }


def create_directory(path, remove_curr):
    if os.path.exists(path):
        if remove_curr:
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def dump_dict(config, path, name='config.csv'):
    w = csv.writer(open(os.path.join(path, name), "w"))
    for key, val in config.items():
        w.writerow([key, val])


def plot_losses(output_file, losses, labels, x_label='steps', y_label='loss'):
    assert(len(losses) == len(labels)), 'mismatch number of losses and labels'
    for (loss, label) in zip(losses, labels):
        plt.plot(loss, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(output_file)
    plt.clf()


def log_aucs(output_file, aucs, names):
    lines = list()
    assert len(names) == len(aucs)
    for auc, name in zip(aucs, names):
        if isinstance(auc, list):
            auc = np.array(auc)
            lines.append(str(name + ':  ' + str(np.mean(auc)) + ' +/- ' + str(np.std(auc))) + '\n')
        else:
            lines.append(str(name + ': ' + str(auc)) + '\n')
    auc_file = open(output_file, 'w+')
    auc_file.writelines(lines)


def random_string(string_length=10):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(string_length))


def indicator_approximation(x, const_1=0.0, const_2=0.001):
    return (const_1 + x) + math.log(const_2 + x) - math.log(const_2)


# def l1_indicator_approximation(x, const_1=0.0, const_2=0.001):
#     return tf.subtract(tf.add(x, tf.math.log(tf.add(x, const_2))), tf.math.log(const_2))
# #
# def l1_indicator_approximation(x, const_1=0.0, const_2=0.001):
#     return tf.add(const_1, x)


def normalize_leaves(leaves, normalize_x=True, normalize_y=True):
    """
    We normalize each feature / target as specified
    """
    if normalize_y:
        y_vals = normalize_sample(np.asarray([leaf.y for leaf in leaves]))
        for i in range(len(leaves)):
            leaves[i].y = y_vals[i]
    if normalize_x:
        for feat_index in range(len(leaves[0].x)):
            feat_vals = normalize_sample(np.asarray([leaf.x[feat_index] for leaf in leaves]))
            for i in range(len(leaves)):
                leaves[i].x[feat_index] = feat_vals[i]


def normalize_sample(sample_array):
    """scaled_val =  (val - mean_val) / sample_variance"""
    z_scores = np.zeros((len(sample_array)))
    mean = np.mean(sample_array)
    std = np.std(sample_array)
    for i in range(len(sample_array)):
        z_scores[i] = (sample_array[i] - mean) / std
    return z_scores
