import os
import json
import jsonpickle
import numpy as np

from utils import random_string, indicator_approximation


class MFNode:
    """
    Multifurcating Node
    """
    def __init__(self, gen_y, x, a, is_leaf=False, parent=None):
        self.parent = parent
        self.is_leaf = is_leaf
        self.descendants = list()
        self.x = x
        self.a = a
        if gen_y is not None:
            self.y = gen_y(x, a)
        self.height = None
        self.name = random_string()
        self.activations = None

    def print_tree(self):
        print(self.x)


class EntangledTree:
    """
    Hard-coding in the number of features and the relationships between them. Future simulations can make an inherited
    class and override the generate_random_child and generate_y functions, as well as the initial values for a

    The relationship is as follows.
    Each specimen has 3 visible features: age (A), caloric intake (B) and gender (C). These 3 visible features generate
    two hidden features: BF% (a combination of A and B, with a non-linearity) and height (a combination of B and C with
    a non-linearity). 4 random variables a0, a1, a2, and a3  effect these relationships. The two hidden variables
    generate the height, which is the y-value, along with another two random variables a4 and a5, and another
    non-linearity.

    At each branch along the tree, the random variables a0-a5 mutate a small random amount. The input features are
    generated completely at random, with no relationship to the parent in the tree.
    """

    def __init__(self, depth=10, mutation_rate=0.01, num_descendants=2, num_leaves=20, low=None, high=None,
                 dendro_loss_mode="l1", seed=0, weights_dim=6):
        self.weights_dim = weights_dim
        self.x_dim = 3
        self.depth = depth
        self.mutation_rate = mutation_rate,
        self.num_descendants = num_descendants
        self.num_leaves = num_leaves
        self.low = low
        self.high = high
        self.seed = seed
        self.tree, self.leaves = self.gen_entangled_tree()
        self.dendro_loss = 0.0
        self.recover_dendro_loss(self.tree, dendro_loss_mode)


    # @staticmethod
    def generate_y(self, x, a):
        # hidden_0 = np.maximum((x[0] * a[0] + x[1] * a[1]), 0)
        # hidden_1 = np.maximum((x[1] * a[2] + x[2] * a[3]), 0)
        # y = np.maximum((hidden_0 * a[4] + hidden_1 * a[5]), 0)
        """
        temporary linear block
        """
        y = x[0] + x[1] + x[2]
        return np.float32(y)

    def gen_entangled_tree(self):
        np.random.seed(self.seed)
        if self.low is not None and self.high is not None:
            starting_a = np.random.uniform(size=self.weights_dim, low=self.low, high=self.high)
        else:
            starting_a = np.random.uniform(size=self.weights_dim)
        starting_x = np.random.normal(loc=1.0, scale=0.5, size=self.x_dim)
        root = MFNode(gen_y=self.generate_y, x=starting_x, a=starting_a)
        curr_depth = 1
        curr_level = list()
        curr_level.append(root)
        leaves = list()
        # internal nodes
        while curr_depth < self.depth - 1:
            next_level = list()
            curr_depth += 1
            for node in curr_level:
                for _ in range(self.num_descendants):
                    new_node = self.generate_random_child(node)
                    next_level.append(new_node)
            curr_level = next_level
        # leaves
        for node in curr_level:
            for _ in range(self.num_leaves):
                new_node = self.generate_random_child(node, is_leaf=True)
                leaves.append(new_node)
        print('Done generating the tree')
        return root, leaves

    def generate_random_child(self, parent_node, is_leaf=False):
        new_x = np.random.normal(loc=1.0, scale=0.5, size=self.x_dim)
        new_a = parent_node.a.copy()
        if not is_leaf:
            for i in range(len(new_a)):
                # new_a[i] = max(0.0, new_a[i] + np.random.normal(scale=self.mutation_rate))
                new_a[i] = new_a[i] + np.random.normal(scale=self.mutation_rate)
        child = MFNode(gen_y=self.generate_y, x=new_x, a=new_a, parent=parent_node, is_leaf=is_leaf)
        parent_node.descendants.append(child)
        return child

    def recover_dendro_loss(self, node, dendro_loss_mode):
        """
        Recursively compute the total dendro loss of the generated tree and store it in self.dendro_loss
        :param node: the root node of the current subtree containing all the data nodes
        :param dendro_loss_mode: method to be used when calculating the loss between nodes (l1 or l2)
        :return: None
        """
        assert dendro_loss_mode in ['indicator_approx', 'l1', 'l2'], 'Unsupported dendro loss mode'
        if node.descendants is not None:
            for child in node.descendants:
                if dendro_loss_mode == 'l1':
                    self.dendro_loss += np.sum(np.abs(np.subtract(node.a, child.a)))
                elif dendro_loss_mode == 'l2':
                    self.dendro_loss += np.sum(np.square(np.subtract(node.a, child.a)))
                elif dendro_loss_mode == 'indicator_approx':
                    for val in np.abs(np.subtract(node.a, child.a)):
                        self.dendro_loss += indicator_approximation(val)
                    # self.dendro_loss += indicator_approximation(np.abs(np.subtract(node.a, child.a)))
                self.recover_dendro_loss(child, dendro_loss_mode)


class BernoulliTree(EntangledTree):

    def __init__(self, depth=10, mutation_rate=0.5, num_descendants=2, num_leaves=20, low=None, high=None,
                 dendro_loss_mode="indicator_approx", mutation_prob=0.1, y_func=0, weights_dim=6,  mutation_style='normal', seed=13, ind_leaves=False):
        assert mutation_style in ['normal', 'exponential']
        np.random.seed(seed)
        self.mutation_style = mutation_style
        self.mutation_prob = mutation_prob
        self.y_func = y_func
        self.leaf_layers = list()
        self.ind_leaves = ind_leaves
        super(BernoulliTree, self).__init__(depth, mutation_rate, num_descendants, num_leaves, low, high,
                 dendro_loss_mode, weights_dim=weights_dim)


    def generate_random_child(self, parent_node, is_leaf=False):
        # new_x = np.asarray(np.random.normal(loc=1.0, scale=0.2, size=self.x_dim), dtype=np.float32)
        # new_x = np.asarray(np.random.uniform(low=0.8, high=5.2, size=self.x_dim), dtype=np.float32)
        new_x = np.asarray([old_x + np.random.normal(loc=0.0, scale=0.25) for old_x in parent_node.x])

        # new_x = np.asarray(np.random.uniform(low=0.8, high=5.2, size=self.x_dim), dtype=np.float32)
        # new_x = np.asarray([1.0 for _ in range(self.x_dim)])
        new_a = parent_node.a.copy()
        if not is_leaf:
            for i in range(len(new_a)):
                if np.random.uniform() < self.mutation_prob:
                    if self.mutation_style == 'normal':
                        new_a[i] = new_a[i] + np.random.normal(scale=self.mutation_rate)
                    elif self.mutation_style == 'exponential':
                        mutation_delta = np.random.exponential(scale=self.mutation_rate)
                        # exponential gives only positive values, need to flip half of them negative
                        if np.random.uniform() < 0.5:
                            mutation_delta *= -1.0
                        new_a[i] = new_a[i] + mutation_delta
        """
        hacking in the independent x-values for leaves quickly
        """
        if is_leaf:
            new_x = np.asarray([np.random.uniform(low=-1.0, high=1.0) for _ in range(len(new_x))])

        child = MFNode(gen_y=self.generate_y, x=new_x, a=new_a, parent=parent_node, is_leaf=is_leaf)
        parent_node.descendants.append(child)
        return child

    def generate_y(self, x, a):

        """
         # this one works! good test without 'a' mutation
        """
        # return np.tanh(x[0] + x[1] + x[2]) + np.random.uniform(low=-0.1, high=0.1)

        """
        A minimally complex generative scheme that encourages disentanglement via sparse mutations
        If the model encapsulates to contribution of a[0] * x[0] in the first hidden node, it can 
        adjust for mutation in a[0] with mutations in only one spot. By splitting the representation across both hidden 
        units, it increases the total number of mutations
        """

        # print('REWRITE Y VALUE')
        # return x[0] + x[1] + x[2]

        hidden_0 = np.tanh(a[0] * (x[0]))
        hidden_1 = np.tanh((a[1] * x[1]))
        hidden_2 = np.tanh(a[2] * x[2])
        # hidden_0 = max((a[0] * (x[0])), 0.0)
        # hidden_1 = max((a[1] * x[1]), 0.0)
        # hidden_2 = max((a[2] * x[2]), 0.0)
        return hidden_0 + hidden_1 + hidden_2 + np.random.normal(scale=0.8)


def recursively_recover_leaves(node, leaf_ids, leaf_list):
    if node.is_leaf:
        # store the node at the correct index
        assert node.name in leaf_ids
        leaf_list[leaf_ids.index(node.name)] = node
    else:
        for child in node.descendants:
            recursively_recover_leaves(child, leaf_ids, leaf_list)


def generate_random_sparse_graph(num_inputs, num_hidden_units, activation_prob=0.5):
    """
    num_inputs: number of input features
    param hidden_units: list containing number of units per hidden layer
    param activation_prob: probability of each connection in the FC graph having a weight other than zero for the generative process
    returns: a dictionary containing all of the layer definitions, integer indicating the total number of weights
    """
    assert num_hidden_units[-1] == 1, 'Incorrect output layer dimension'
    layers = dict()
    total_weights = 0  # note that this does not account for bias terms
    #keep adding hidden layers, randomly selecting outputs from the previous layer to use as input
    prev_outputs = range(num_inputs)
    for i in range(len(num_hidden_units)):  # for each hidden layer
        nodes_activated = set()
        layers[i] = dict()  # each key is an output, value is list of inputs it takes

        for output in range(num_hidden_units[i]):
            layers[i][output] = list()
            for input in prev_outputs:
                if np.random.uniform() < activation_prob:
                    layers[i][output].append(input)
                    total_weights += 1
                    nodes_activated.add(output)
        if len(nodes_activated) == 0:  # randomly pick at least one connection
            input = np.random.choice(prev_outputs)
            output = np.random.choice(list(range(num_hidden_units[i])))
            layers[i][output].append(input)  # defines an active connection
            nodes_activated.add(output)
            total_weights += 1
        prev_outputs = list(nodes_activated)

    return layers, total_weights


def generate_y_from_sparse_graph(inputs, input_weights, graph_def, num_hidden_units, activation='tanh', add_noise=True):
    """
    param inputs:
    param input_weights:
    param graph_def:
    param num_hidden_units:
    param activation:
    param add_noise:
    return: y-value created by the inputs and sparse causal graph, list of activations from all nodes
    """
    assert activation in ['tanh', 'relu', 'none'], 'Unsupported activation function when generating y'
    assert num_hidden_units[-1] == 1, 'Incorrect output layer dimension'
    weight_index = 0
    activations = list()
    curr_inputs = inputs
    for i in range(len(num_hidden_units)):
        layer_def = graph_def[i]
        layer_outputs = [0 for _ in range(num_hidden_units[i])]
        for node_idx in range(num_hidden_units[i]):
            if node_idx in layer_def.keys():
                output = 0.0
                # weighted sum
                for input_idx in layer_def[node_idx]:
                    output += input_weights[weight_index] * curr_inputs[input_idx]
                    weight_index += 1
                # activation function
                if activation == 'tanh':
                    output = np.tanh(output)
                elif activation == 'relu':
                    output = max(0.0, output)
                #store the output for use by the next layer, and in activations for later analysis
                layer_outputs[node_idx] = output
                activations.append(output)
        curr_inputs = layer_outputs

    # at the very end, curr_inputs should hold the single y-value
    assert len(curr_inputs) == num_hidden_units[-1]
    y = curr_inputs[0]
    if add_noise:
        y += np.random.normal(loc=0, scale=0.001)
    return y, activations


def store_tree_and_leaves(tree, leaves, folder_name, tree_basename='tree.json', leaf_basename='leaf.json'):
    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, tree_basename), 'w') as out:
        json.dump(jsonpickle.encode(tree), out, indent=4)
    out.close()
    leaf_dict = {'leaves': [leaf.name for leaf in leaves]}
    with open(os.path.join(folder_name, leaf_basename), 'w') as out:
        json.dump(leaf_dict, out, indent=4)
    out.close()


def load_tree_and_leaves(folder_name, tree_basename='tree.json', leaf_basename='leaf.json'):
    with open(os.path.join(folder_name, tree_basename)) as file:
        js_string = json.load(file)
    file.close()
    tree = jsonpickle.decode(js_string)
    with open(os.path.join(folder_name, leaf_basename)) as file:
        leaf_dict = json.load(file)
    file.close()
    leaf_ids = leaf_dict['leaves']
    leaves = [MFNode(None, None, None) for _ in range(len(leaf_ids))]
    recursively_recover_leaves(tree, leaf_ids, leaves)
    return tree, leaves

# testing the data saving and loading functionality
# test_tree = EntangledTree(mutation_rate=0.1, depth=8, num_leaves=1, low=0.0, high=5.0)
# store_tree_and_leaves(test_tree.tree, test_tree.leaves, 'tree_storage/seed0')
# loaded_tree, loaded_leaves = load_tree_and_leaves('tree_storage/seed0')

# layers, total_weights = generate_random_sparse_graph(3, [3, 3, 3, 1])
# generate_y_from_sparse_graph(inputs=[3.0, 3.0, 3.0], input_weights=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], graph_def=layers, num_hidden_units=[3, 3, 3, 1])
# print('done')
