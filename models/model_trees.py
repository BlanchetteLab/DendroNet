from abc import ABC

import tensorflow as tf
from tensorflow.keras import Model
from utils import tf_indicator_approximation, indicator_approximation
from data_structures.gen_tree import MultifurcatingLayerNode
from models.custom_models import FeedForwardSubModel, EntangledModel, LinRegModel


"""
Writing a new parent class, using the multifurcating node structure, able to be sub-classed for specific model 
architectures. TreeModelNN should become a subclass of this.
"""
class TreeModelMF(Model):
    def __init__(self, data_tree_root, leaves, model_spec_dict, dummy_input,
                 min_distance = 1e-7, delta_loss_mode='l1'):
        print(str(len(leaves)))
        super(TreeModelMF, self).__init__()
        layers = list()
        self.data_tree_root = data_tree_root
        self.leaves = leaves
        self.model_spec_dict = model_spec_dict
        self.dummy_input = dummy_input
        self.min_distance = min_distance
        self.delta_loss_mode=delta_loss_mode
        self.delta_tensor = tf.identity([0.0])
        self.root_model = self.instantiate_submodel()
        self.root_layer = MultifurcatingLayerNode(self.root_model)
        self.num_weight_tensors = len(self.root_model.weights)
        layers.append(self.root_layer)
        self.leaf_layers = list()
        self.copy_child_nodes(data_tree_root, self.root_layer, layers, leaves)

        # initialize all the layers to be attributes, allows for the tracking of gradients
        for i in range(len(layers)):
            setattr(self, str('layer_'+str(i)), layers[i].layer)

    """
    By default, this parent class will implement linear regression. Subclasses will implement different submodels
    by overriding this function, use requires including the proper elements in the model_spec_dict passed in by
    the experiment file
    """
    def instantiate_submodel(self):
        submodel = LinRegModel(self.model_spec_dict['layer_shape'])
        # submodel.build(self.model_spec_dict['layer_shape'])
        submodel.call(self.dummy_input)  # causes the weights to be instantiated to real values
        return submodel

    def copy_child_nodes(self, data_node, layer_node, layers, leaves):
        layer_node.height = data_node.height
        # if node is a leaf store the index of the corresponding training example
        if data_node.descendants is None or len(data_node.descendants) == 0:
            layer_node.is_leaf = True
            found_match = False
            for i in range(len(leaves)):
                if leaves[i] == data_node:
                    layer_node.train_index = i
                    self.leaf_layers.append(layer_node)
                    found_match = True
            if not found_match:
                print('missing leaf match')
            submodel = self.instantiate_submodel()
            layer_node.layer = submodel

        # if not a leaf, recursively copy the children
        else:
            for data_child in data_node.descendants:
                new_layer = self.instantiate_submodel()
                layer_node.descendants.append(MultifurcatingLayerNode(layer=new_layer, parent=layer_node))
                layers.append(layer_node.descendants[-1])
                self.copy_child_nodes(data_child, layer_node.descendants[-1], layers, leaves)

    def calc_delta_loss(self, layernode):
        curr_weights = layernode.layer.weights
        prev_weights = layernode.parent.layer.weights
        if layernode.height is None or layernode.parent.height is None:
            for i in range(self.num_weight_tensors):
                self.delta_tensor = tf.add(self.delta_tensor, tf.reduce_sum(tf.abs(tf.subtract(curr_weights[i], prev_weights[i]))))
        else:
            # clamp distance to the configured min distance, prevents dividing by zero, excessive penalties etc.
            mut_distance = tf.math.maximum(tf.abs(tf.subtract(
                layernode.height, layernode.parent.height)), self.min_distance)
            for i in range(self.num_weight_tensors):
                new_delta = tf.math.divide(
                    tf.reduce_sum(tf.abs(tf.subtract(curr_weights[i], prev_weights[i]))), mut_distance)
                self.delta_tensor = tf.add(self.delta_tensor, new_delta)

    def call_child_layers(self, layernode, x):
        # curr_weights = layernode.layer.weights
        # adding delta loss term
        if layernode.parent is not None:
            self.calc_delta_loss(layernode)

        # make actual prediction if node is a leaf
        if layernode.is_leaf:
            curr_x = x[layernode.train_index:(layernode.train_index + 1)]
            return layernode.layer.call(curr_x)

        # otherwise keep recursively passing weights down the tree
        if len(layernode.descendants) == 1:
            return tf.identity(self.call_child_layers(layernode.descendants[0], x))
        else:
            return tf.concat([self.call_child_layers(child, x) for child in layernode.descendants],
                             axis=0)

    # this is the function that gets called when we pass inputs into an instance of the model
    def call(self, x):
        self.delta_tensor = tf.identity([0.0])  # need to reset the delta to zero at every call
        result = self.call_child_layers(self.root_layer, x)
        return result, self.delta_tensor


class TreeModelMFEntangledNN(TreeModelMF):
    def __init__(self, data_tree_root, leaves, model_spec_dict, dummy_input,
                 min_distance = 1e-7, delta_loss_mode='l1'):
        print(str(len(leaves)))
        self.delta_constant = tf.constant(1.0)
        super(TreeModelMFEntangledNN, self).__init__(data_tree_root, leaves, model_spec_dict, dummy_input, min_distance,
                                            delta_loss_mode)

    def instantiate_submodel(self):
        net = EntangledModel(self.model_spec_dict['output_dim'], self.model_spec_dict['num_dense_units'],
                                  (self.model_spec_dict['num_features'],))
        net.call(self.dummy_input)
        return net

    """
    We want a loss function that encourages the fewest number of mutations as possible. We approximate this
    by using log(x + 1) for each abs difference between weights x. We can scale x by a scaling factor if
    we want to increase the slope. Key points: the slope increases the closer we get to zero. The loss for is zero
    for a mutation value of zero. The sum of the loss of two small mutations 0.5m is larger than one the loss of one 
    mutation of size m (this should encourage disentanglement) 
    """
    # todo: add the scaling factor
    # todo: try and change this to a tf.scan instead of a for loop
    def calc_delta_loss(self, layernode):
        curr_weights = layernode.layer.weights
        prev_weights = layernode.parent.layer.weights
        for i in range(self.num_weight_tensors):
            self.delta_tensor = tf.add(self.delta_tensor, tf.reduce_sum(
                tf.math.log(tf.add(tf.abs(tf.subtract(curr_weights[i], prev_weights[i])), self.delta_constant))))


class TreeModelMFFeedforwardNN(TreeModelMF):
    def __init__(self, data_tree_root, leaves, model_spec_dict, dummy_input,
                 min_distance = 1e-7, delta_loss_mode='l1'):
        print(str(len(leaves)))
        super(TreeModelMFFeedforwardNN, self).__init__(data_tree_root, leaves, model_spec_dict, dummy_input, min_distance,
                                            delta_loss_mode)

    def instantiate_submodel(self):
        net = FeedForwardSubModel(self.model_spec_dict['output_dim'], self.model_spec_dict['num_dense_units'],
                                  (self.model_spec_dict['num_features'],))
        net.call(self.dummy_input)
        return net
