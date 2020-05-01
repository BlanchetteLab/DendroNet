import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from models.custom_layers import AdditionLayer
from data_structures.entangled_data_simulation import BernoulliTree

class DendroLinNode:
    def __init__(self, weights, parent=None, is_leaf=False):
        self.weights = weights
        self.children = list()
        self.is_leaf = is_leaf
        self.parent = parent
        self.weight_list = list()


class DendroLinReg(Model):
    def __init__(self, data_root, input_dim):
        """
        :param data_root: root node of the data tree
        :param layer_shape: assumes a bias term is already appended to the feature vectors
        """
        super(DendroLinReg, self).__init__()
        self.data_root = data_root
        self.input_dim = input_dim
        self.leaf_list = list()
        self.delta_tensor = tf.identity([0.0])
        # todo: change these to small random numbers?
        self.root_weights = tf.Variable(tf.identity([0.0 for _ in range(self.input_dim)]), trainable=True)
        self.model_root = DendroLinNode(weights=self.root_weights)
        self.construct_tree(self.model_root, self.data_root)

    def construct_tree(self, model_node, data_node):
        for data_child in data_node.descendants:
            new_weights = tf.Variable(tf.identity([0.0 for _ in range(self.input_dim)]), trainable=True)
            model_child = DendroLinNode(weights=new_weights, parent=model_node, is_leaf=data_child.is_leaf)
            model_node.children.append(model_child)
            self.construct_tree(model_child, data_child)

        """
        concatenating relevant parent tf.Variables, so that they can be accessed during call function for a leaf
        """
        model_node.weight_list.append(model_node.weights)
        curr_parent = model_node.parent
        while curr_parent is not None:
            model_node.weight_list.append(curr_parent.weights)
            curr_parent = curr_parent.parent
        if model_node.is_leaf:
            self.leaf_list.append(model_node)

    @tf.function
    def calc_delta_loss(self):
        self.delta_tensor = tf.identity([0.0])
        self.recursively_sum_deltas(self.model_root)
        return self.delta_tensor

    def recursively_sum_deltas(self, node):
        for child in node.children:
            self.delta_tensor = tf.add(self.delta_tensor, tf.reduce_sum(tf.abs(tf.subtract(node.weights, child.weights))))
            self.recursively_sum_deltas(child)

    @tf.function
    def call(self, x, leaf):
        weights = tf.expand_dims(tf.reduce_sum(leaf.weight_list, axis=0), axis=0)
        #todo: move these x transformations into preprocessing
        return tf.matmul(tf.expand_dims(tf.identity(x), axis=0), tf.transpose(weights))

num_steps = 1000
dpf = 0.01
if __name__ == '__main__':
    data_tree = BernoulliTree(mutation_rate=0.0, depth=12, num_leaves=2, low=0.0, high=1.0,
                              mutation_prob=0.0, mutation_style='exponential')

    data_root = data_tree.tree
    # todo: add bias term to x vectors
    tree_model = DendroLinReg(data_root, 4)
    leaves = tree_model.leaf_list
    print(str(len(leaves)) + ' num leaves')
    """
    preparing the data for training loop
    """
    x = list()
    y = list()
    for leaf in data_tree.leaves:
        leaf.x = np.append(leaf.x, 1.0)  # bias term
        x.append(leaf.x)
        y.append(leaf.y)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)


    """
    training loop
    """

    result = tree_model.call(x[0], leaves[0])
    print(result)
    delta_loss = tree_model.calc_delta_loss()
    print(delta_loss)
    optimizer = tf.optimizers.Adam()
    for step in range(num_steps):
        with tf.GradientTape() as tape:
            # y_hat = [tree_model.call(x[i], leaves[i]) for i in range(len(leaves))]
            y_hat = tf.stack([tf.squeeze(tree_model.call(x[i], leaves[i])) for i in range(len(leaves))], axis=0)
            regression_loss = tf.losses.mean_squared_error(y_hat, y)
            delta_loss = tree_model.calc_delta_loss() * dpf
            loss = regression_loss + delta_loss
            # loss = regression_loss
            gradients = tape.gradient(loss, tree_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, tree_model.trainable_variables))
            print('loss: ' + str(loss))
            print('delta loss: ' + str(delta_loss))

    # @tf.function
    # def tree_train_step(self, optimizer, dendro_reg=True):
    #     with tf.GradientTape() as tape:
    #         if self.config['l2']:
    #             y_hat, delta, l2 = self.tree_model(self.all_x)
    #         else:
    #             y_hat, delta = self.tree_model(self.all_x)
    #         y_hat_train = tf.gather(y_hat, self.train_idx)
    #         y_hat_valid = tf.gather(y_hat, self.valid_idx)
    #         dendronet_loss = self.config['delta_penalty_factor'] * delta
    #         predict_loss = self.loss_function(self.train_y, y_hat_train) * self.config['loss_scale']
    #         if dendro_reg:
    #             loss = predict_loss + dendronet_loss
    #         else:
    #             loss = predict_loss
    #         if self.config['l2']:
    #             l2_loss = l2 * self.config['l2_penalty_factor'] # todo: make the l2 penalty factor scale with the number of nodes
    #             loss = loss + l2_loss
    #         valid_loss = self.loss_function(self.valid_y, y_hat_valid)
    #     gradients = tape.gradient(loss, self.tree_model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, self.tree_model.trainable_variables))
    #     if self.calc_auc:
    #         self.dendro_auc.update_state(self.valid_y[:,0:1], y_hat_valid[:,0:1])
    #     return tf.reduce_mean(predict_loss), dendronet_loss, tf.reduce_mean(valid_loss)
