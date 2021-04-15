import tensorflow as tf
from tensorflow.keras import Model
from data_structures.gen_tree import LayerNode, MultifurcatingLayerNode
from models.custom_layers import AdditionLayer, MatmulLayer


class TreeModel(Model):
    def __init__(self, data_tree_root, leaves, weights_dim, min_distance = 1e-7):
        print(str(len(leaves)))
        super(TreeModel, self).__init__()
        layers = list()
        self.min_distance = min_distance
        self.weights_dim = weights_dim
        self.addition_layer_0 = AdditionLayer(self.weights_dim)
        self.root_layer = LayerNode(self.addition_layer_0)
        # replicate the data-tree structure in layers
        self.copy_child_nodes(data_tree_root, self.root_layer, layers, leaves)
        self.zero_vector = tf.identity([0.0] * weights_dim)
        self.delta_tensor = tf.identity([0.0])
        # initialize all the layers to be attributes for god knows what reason
        for i in range(len(layers)):
            setattr(self, str(i), layers[i].layer)
            layers[i].layer.build(self.weights_dim)

    def copy_child_nodes(self, data_node, layer_node, layers, leaves):
        layer_node.height = data_node.height
        # if node is a leaf store the index of the corresponding training example
        if data_node.left is None and data_node.right is None:
            layer_node.is_leaf = True
            # may be able to remove this if we are careful about preserving order
            for i in range(len(leaves)):
                if leaves[i] == data_node:
                    layer_node.train_index = i
            """
            Note that we could eventually store a list of parents here and get a path of the layers that need to be
            fired to predict this leaf, but for now we assume we always fire on the entire tree at once, as this 
            actually reduces total number of computations for a training epoch and makes calculating the regularization
            error much simpler
            """
        # if not a leaf, recursively copy the children
        else:
            if data_node.left is not None:
                left_layer = AdditionLayer(self.weights_dim)
                layer_node.left = LayerNode(layer=left_layer, parent=layer_node)
                layers.append(layer_node.left)
                self.copy_child_nodes(data_node.left, layer_node.left, layers, leaves)
            if data_node.right is not None:
                right_layer = AdditionLayer(self.weights_dim)
                layer_node.right = LayerNode(layer=right_layer, parent=layer_node)
                layers.append(layer_node.right)
                self.copy_child_nodes(data_node.right, layer_node.right, layers, leaves)

    def call_child_layers(self, layernode, input_weights, x):
        curr_weights = layernode.layer(input_weights)
        if layernode.height is None or layernode.parent is None or layernode.parent.height is None:
            self.delta_tensor = tf.add(self.delta_tensor, tf.reduce_sum(tf.math.squared_difference(curr_weights, input_weights)))

        else:
            # clamp distance to the configured min distance, prevents dividing by zero, excessive penalties etc.
            mut_distance = tf.math.maximum(tf.subtract(layernode.height, layernode.parent.height), self.min_distance)
            new_delta = tf.math.divide(
                tf.reduce_sum(tf.math.squared_difference(curr_weights, input_weights)), mut_distance)
            self.delta_tensor = tf.add(self.delta_tensor, new_delta)
        # make actual prediction if node is a leaf
        if layernode.is_leaf:
            curr_x = x[layernode.train_index:(layernode.train_index + 1)]

            return tf.expand_dims(tf.reduce_sum(tf.matmul(curr_x, tf.transpose(tf.expand_dims(curr_weights, 0)))), 0)
        # otherwise keep recursively passing weights down the tree
        elif layernode.left is not None and layernode.right is not None:
            return tf.concat([self.call_child_layers(layernode.left, curr_weights, x),
                              self.call_child_layers(layernode.right, curr_weights, x)], axis=0)

    # this is the function that gets called when we pass inputs into an instance of the model
    def call(self, x):
        self.delta_tensor = tf.identity([0.0]) # need to reset the delta to zero at every call
        result = self.call_child_layers(self.root_layer, self.zero_vector, x)
        return result, self.delta_tensor


# class LinRegModel(Model):
#     def __init__(self, weights_dim):
#         super(LinRegModel, self).__init__()
#         self.weights_dim = weights_dim
#         self.matmul_layer = MatmulLayer(self.weights_dim)
#         self.matmul_layer.build(self.weights_dim)
#
#     def call(self, x):
#         result = self.matmul_layer.call(x)
#         return tf.reduce_sum(result, axis=1)

class LinRegModel(Model):
    def __init__(self, layer_shape):
        super(LinRegModel, self).__init__()
        self.layer_shape = layer_shape
        self.input_dim = layer_shape[-2]
        self.output_dim = self.layer_shape[-1]
        self.matmul_layer = MatmulLayer(self.layer_shape)
        self.matmul_layer.build(self.layer_shape)

    def call(self, x):
        x = tf.reshape(x, (-1, self.input_dim, self.output_dim))
        # result = tf.squeeze(tf.concat(tf.map_fn(
        #     lambda x_i: self.matmul_layer.call(
        #         tf.transpose(tf.expand_dims(x_i, 0))), x), axis=0), axis=1)
        result = self.matmul_layer.call(x)
        # return tf.reduce_sum(result, axis=1)
        return result


# class LogRegModel(Model):
#     def __init__(self, layer_shape):
#         super(LogRegModel, self).__init__()
#         self.layer_shape = layer_shape
#         self.matmul_layer = MatmulLayer(self.layer_shape)
#         self.matmul_layer.build(self.layer_shape)
#
#     def call(self, x):
#         result = self.matmul_layer.call(x)
#         return tf.nn.softmax(result)

class LogRegModel(Model):
    def __init__(self, layer_shape):
        super(LogRegModel, self).__init__()
        self.layer_shape = layer_shape
        self.input_dim = layer_shape[-2]
        self.output_dim = self.layer_shape[-1]
        self.matmul_layer = MatmulLayer(self.layer_shape)
        self.matmul_layer.build(self.layer_shape)

    def call(self, x):
        x = tf.reshape(x, (-1, self.input_dim, 1))
        # result = tf.squeeze(tf.concat(tf.map_fn(
        #     lambda x_i: self.matmul_layer.call(
        #         tf.transpose(tf.expand_dims(x_i, 0))), x), axis=0), axis=1)
        result = self.matmul_layer.call(x)
        # return tf.reduce_sum(result, axis=1)
        # return tf.nn.softmax(result)
        return tf.nn.softmax(tf.squeeze(result, axis=-1))



# todo: make a general tree  model that accepts the prediction function as argument, 90% of this is re-used from above
class TreeModelLogReg(Model):
    def __init__(self, data_tree_root, leaves, layer_shape, min_distance = 1e-7):
        print(str(len(leaves)))
        super(TreeModelLogReg, self).__init__()
        layers = list()
        self.min_distance = min_distance
        self.layer_shape = layer_shape
        self.addition_layer_0 = AdditionLayer(self.layer_shape)
        self.root_layer = LayerNode(self.addition_layer_0)
        # replicate the data-tree structure in layers
        self.copy_child_nodes(data_tree_root, self.root_layer, layers, leaves)
        self.zero_vector = tf.zeros(layer_shape)
        self.delta_tensor = tf.identity([0.0])
        # initialize all the layers to be attributes for god knows what reason
        for i in range(len(layers)):
            setattr(self, str(i), layers[i].layer)
            layers[i].layer.build(self.layer_shape)

    def copy_child_nodes(self, data_node, layer_node, layers, leaves):
        layer_node.height = data_node.height
        # if node is a leaf store the index of the corresponding training example
        if data_node.left is None and data_node.right is None:
            layer_node.is_leaf = True
            # may be able to remove this if we are careful about preserving order
            for i in range(len(leaves)):
                if leaves[i] == data_node:
                    layer_node.train_index = i
            """
            Note that we could eventually store a list of parents here and get a path of the layers that need to be
            fired to predict this leaf, but for now we assume we always fire on the entire tree at once, as this 
            actually reduces total number of computations for a training epoch and makes calculating the regularization
            error much simpler
            """
        # if not a leaf, recursively copy the children
        else:
            if data_node.left is not None:
                left_layer = AdditionLayer(self.layer_shape)
                layer_node.left = LayerNode(layer=left_layer, parent=layer_node)
                layers.append(layer_node.left)
                self.copy_child_nodes(data_node.left, layer_node.left, layers, leaves)
            if data_node.right is not None:
                right_layer = AdditionLayer(self.layer_shape)
                layer_node.right = LayerNode(layer=right_layer, parent=layer_node)
                layers.append(layer_node.right)
                self.copy_child_nodes(data_node.right, layer_node.right, layers, leaves)

    def call_child_layers(self, layernode, input_weights, x):
        curr_weights = layernode.layer(input_weights)
        if layernode.height is None or layernode.parent is None or layernode.parent.height is None:
            self.delta_tensor = tf.add(self.delta_tensor, tf.reduce_sum(tf.math.squared_difference(curr_weights, input_weights)))

        else:
            # clamp distance to the configured min distance, prevents dividing by zero, excessive penalties etc.
            mut_distance = tf.math.maximum(tf.subtract(layernode.height, layernode.parent.height), self.min_distance)
            # new_delta = tf.math.divide(
            #     tf.reduce_sum(tf.math.squared_difference(curr_weights, input_weights)), mut_distance)
            new_delta = tf.math.divide(
                tf.reduce_sum(tf.abs(tf.subtract(curr_weights, input_weights))), mut_distance)
            self.delta_tensor = tf.add(self.delta_tensor, new_delta)
        # make actual prediction if node is a leaf
        if layernode.is_leaf:
            curr_x = x[layernode.train_index:(layernode.train_index + 1)]
            # todo: this is the part where we should be taking in an argument, its the only line that changes
            return tf.nn.softmax(tf.matmul(curr_x, curr_weights))
        # otherwise keep recursively passing weights down the tree
        elif layernode.left is not None and layernode.right is not None:
            return tf.concat([self.call_child_layers(layernode.left, curr_weights, x),
                              self.call_child_layers(layernode.right, curr_weights, x)], axis=0)
        elif layernode.left is not None:
            return tf.identity(self.call_child_layers(layernode.left, curr_weights, x))
        else:
            return tf.identity(self.call_child_layers(layernode.right, curr_weights, x))

    # this is the function that gets called when we pass inputs into an instance of the model
    def call(self, x):
        self.delta_tensor = tf.identity([0.0]) # need to reset the delta to zero at every call
        result = self.call_child_layers(self.root_layer, self.zero_vector, x)
        return result, self.delta_tensor


# todo: see if this can just replace the binary tree model completely
class MultifurcatingTreeModelLogReg(Model):
    def __init__(self, data_tree_root, leaves, layer_shape, min_distance = 1e-7):
        print(str(len(leaves)))
        super(MultifurcatingTreeModelLogReg, self).__init__()
        layers = list()
        self.min_distance = min_distance
        self.layer_shape = layer_shape
        self.addition_layer_0 = AdditionLayer(self.layer_shape)
        self.root_layer = MultifurcatingLayerNode(self.addition_layer_0)
        # replicate the data-tree structure in layers
        self.copy_child_nodes(data_tree_root, self.root_layer, layers, leaves)
        self.zero_vector = tf.zeros(layer_shape)
        self.delta_tensor = tf.identity([0.0])
        # initialize all the layers to be attributes for god knows what reason
        for i in range(len(layers)):
            setattr(self, str(i), layers[i].layer)
            layers[i].layer.build(self.layer_shape)

    def copy_child_nodes(self, data_node, layer_node, layers, leaves):
        layer_node.height = data_node.height
        # if node is a leaf store the index of the corresponding training example
        if data_node.descendants is None or len(data_node.descendants) == 0:
            layer_node.is_leaf = True
            # may be able to remove this if we are careful about preserving order
            found_match = False
            for i in range(len(leaves)):
                if leaves[i] == data_node:
                    layer_node.train_index = i
                    found_match = True
            # assert found_match, 'Missing leaf match'
            if not found_match:
                print('missing leaf match')
            """
            Note that we could eventually store a list of parents here and get a path of the layers that need to be
            fired to predict this leaf, but for now we assume we always fire on the entire tree at once, as this 
            actually reduces total number of computations for a training epoch and makes calculating the regularization
            error much simpler
            """
        # if not a leaf, recursively copy the children
        else:
            for data_child in data_node.descendants:
                new_layer = AdditionLayer(self.layer_shape)
                layer_node.descendants.append(MultifurcatingLayerNode(layer=new_layer, parent=layer_node))
                layers.append(layer_node.descendants[-1])
                self.copy_child_nodes(data_child, layer_node.descendants[-1], layers, leaves)

    def call_child_layers(self, layernode, input_weights, x):
        curr_weights = layernode.layer(input_weights)
        if layernode.height is None or layernode.parent is None or layernode.parent.height is None:
            self.delta_tensor = tf.add(self.delta_tensor, tf.reduce_sum(tf.math.squared_difference(curr_weights, input_weights)))
        else:
            # clamp distance to the configured min distance, prevents dividing by zero, excessive penalties etc.
            mut_distance = tf.math.maximum(tf.abs(tf.subtract(layernode.height, layernode.parent.height)), self.min_distance)
            # new_delta = tf.math.divide(
            #     tf.reduce_sum(tf.math.squared_difference(curr_weights, input_weights)), mut_distance)
            new_delta = tf.math.divide(
                tf.reduce_sum(tf.abs(tf.subtract(curr_weights, input_weights))), mut_distance)
            self.delta_tensor = tf.add(self.delta_tensor, new_delta)
        # make actual prediction if node is a leaf
        if layernode.is_leaf:
            curr_x = x[layernode.train_index:(layernode.train_index + 1)]
            # todo: this is the part where we should be taking in an argument, its the only line that changes
            return tf.nn.softmax(tf.matmul(curr_x, curr_weights))
        # otherwise keep recursively passing weights down the tree
        assert len(layernode.descendants) > 0
        if len(layernode.descendants) == 1:
            return tf.identity(self.call_child_layers(layernode.descendants[0], curr_weights, x))
        else:
            return tf.concat([self.call_child_layers(child, curr_weights, x) for child in layernode.descendants],
                             axis=0)

    # this is the function that gets called when we pass inputs into an instance of the model
    def call(self, x):
        self.delta_tensor = tf.identity([0.0]) # need to reset the delta to zero at every call
        result = self.call_child_layers(self.root_layer, self.zero_vector, x)
        return result, self.delta_tensor



class MultifurcatingTreeModelLinReg(MultifurcatingTreeModelLogReg):
    def __init__(self, data_tree_root, leaves, layer_shape, min_distance = 1e-7):
        print(str(len(leaves)))
        super(MultifurcatingTreeModelLinReg, self).__init__(data_tree_root, leaves, layer_shape, min_distance = 1e-7)
        layers = list()
        self.min_distance = min_distance
        self.layer_shape = layer_shape
        self.addition_layer_0 = AdditionLayer(self.layer_shape)
        self.root_layer = MultifurcatingLayerNode(self.addition_layer_0)
        # replicate the data-tree structure in layers
        self.copy_child_nodes(data_tree_root, self.root_layer, layers, leaves)
        self.zero_vector = tf.zeros(layer_shape)
        self.delta_tensor = tf.identity([0.0])
        # initialize all the layers to be attributes for god knows what reason
        for i in range(len(layers)):
            setattr(self, str(i), layers[i].layer)
            layers[i].layer.build(self.layer_shape)


class FeedForwardSubModel(tf.keras.Model):
    def __init__(self, num_classes, num_weights, input_shape):
        super(FeedForwardSubModel, self).__init__()
        self.layer_0 = tf.keras.layers.Dense(num_weights, activation='tanh', input_shape=input_shape)
        self.layer_1 = tf.keras.layers.Dense(num_weights, activation='tanh')
        # self.layer_2 = tf.keras.layers.Dense(num_classes, activation='softmax')
        self.layer_2 = tf.keras.layers.Dense(num_classes)

    def call(self, inputs):
        return self.layer_2(self.layer_1(self.layer_0(tf.expand_dims(inputs, axis=0))))


class EntangledModel(tf.keras.Model):
    def __init__(self, num_classes, num_weights, input_shape):
        super(EntangledModel, self).__init__()
        self.layer_0 = tf.keras.layers.Dense(num_weights, activation='tanh', input_shape=input_shape)
        self.layer_1 = tf.keras.layers.Dense(num_classes)

    def call(self, inputs):
        return self.layer_1(self.layer_0(tf.expand_dims(inputs, axis=0)))


class RandomSparseModel(tf.keras.Model):
    """
    corresponds to hidden unit structure of [num_weights, num_weights, num_classes]
    """
    def __init__(self, num_classes, num_weights, input_shape):
        super(RandomSparseModel, self).__init__()
        self.layer_0 = tf.keras.layers.Dense(num_weights, activation='tanh', input_shape=input_shape)
        self.layer_1 = tf.keras.layers.Dense(num_weights, activation='tanh')
        self.layer_2 = tf.keras.layers.Dense(num_classes)

    def call(self, inputs):
        return self.layer_2(self.layer_1(self.layer_0(tf.expand_dims(inputs, axis=0))))

    def get_activations(self, inputs):
        activations = tf.zeros(shape=(1,))
        layer_0_out = self.layer_0(tf.expand_dims(inputs, axis=0))
        activations = tf.concat((activations, tf.squeeze(layer_0_out)), axis=0)
        layer_1_out = self.layer_1(layer_0_out)
        activations = tf.concat((activations, tf.squeeze(layer_1_out)), axis=0)
        layer_2_out = self.layer_2(layer_1_out)
        activations = tf.concat((activations, tf.reshape(layer_2_out, shape=(1,))), axis=0)
        return activations[1:]  # discarding the dummy zero from the declaration of the tensor


class FeedForwardClassification(tf.keras.Model):
    """
    corresponds to hidden unit structure of [num_weights, num_weights, num_classes]
    """
    def __init__(self, num_classes, num_weights, input_shape):
        super(FeedForwardClassification, self).__init__()
        self.layer_0 = tf.keras.layers.Dense(num_weights, activation='relu', input_shape=input_shape)
        self.layer_1 = tf.keras.layers.Dense(num_weights, activation='relu', input_shape=(1, num_weights))
        self.layer_2 = tf.keras.layers.Dense(num_classes, activation='softmax', input_shape=(1, num_weights))

    def call(self, inputs):
        # return self.layer_2(self.layer_1(self.layer_0(tf.expand_dims(inputs, axis=0))))
        # return self.layer_2(self.layer_1(self.layer_0(inputs)))
        return self.layer_2(self.layer_1(self.layer_0(tf.expand_dims(tf.squeeze(inputs, axis=1), axis=0))))


# class MinEntangledModel(tf.keras.Model):
#     def __init__(self, num_classes, num_weights, input_shape):
#         super(MinEntangledModel, self).__init__()
#         self.layer_0 = tf.keras.layers.Dense(num_classes, activation='tanh', input_shape=input_shape)
#         self.layer_1 = tf.keras.layers.Dense(num_classes)
#
#     def call(self, inputs):
#         return self.layer_1(self.layer_0(tf.expand_dims(inputs, axis=0)))

class NotTrainableFeedForwardSubModel(tf.keras.Model):
    def __init__(self, num_classes, num_weights, input_shape):
        super(NotTrainableFeedForwardSubModel, self).__init__()
        self.layer_0 = tf.keras.layers.Dense(num_weights, activation='relu', trainable=False, input_shape=input_shape)
        self.layer_1 = tf.keras.layers.Dense(num_weights, activation='relu', trainable=False)
        self.layer_2 = tf.keras.layers.Dense(num_classes, activation='softmax', trainable=False)

    def call(self, inputs):
        return self.layer_2(self.layer_1(self.layer_0(inputs)))


class TreeModelNN(Model):
    def __init__(self, data_tree_root, leaves, layer_shape, output_dim, num_dense_units, num_features, splice_sites,
                 weight_shapes, min_distance = 1e-7):
        print(str(len(leaves)))
        super(TreeModelNN, self).__init__()
        layers = list()
        self.min_distance = min_distance
        self.layer_shape = layer_shape
        self.num_features = num_features
        self.addition_layer_0 = AdditionLayer(self.layer_shape)
        self.root_layer = MultifurcatingLayerNode(self.addition_layer_0)
        self.zero_vector = tf.zeros(layer_shape)
        self.delta_tensor = tf.identity([0.0])
        self.dummy_input = tf.zeros(shape=(1, num_features))
        self.output_dim = output_dim
        self.num_dense_units = num_dense_units
        self.splice_sites = splice_sites
        self.weight_shapes = weight_shapes
        # replicate the data-tree structure in layers
        self.copy_child_nodes(data_tree_root, self.root_layer, layers, leaves)

        # initialize all the layers to be attributes for god knows what reason
        for i in range(len(layers)):
            setattr(self, str(i), layers[i].layer)
            layers[i].layer.build(self.layer_shape)
            if layers[i].is_leaf:
                setattr(self, str(i) + '_model', layers[i].model)

    def retrieve_nn_weights(self, weight_tensor):
        weights = list()
        for i in range(1, len(self.splice_sites)):
            weights.append(tf.identity(weight_tensor[self.splice_sites[i-1]:self.splice_sites[i]]))
        for i in range(len(self.weight_shapes)):
            weights[i] = tf.reshape(weights[i], self.weight_shapes[i])
        return weights

    def copy_child_nodes(self, data_node, layer_node, layers, leaves):
        layer_node.height = data_node.height
        # if node is a leaf store the index of the corresponding training example
        if data_node.descendants is None or len(data_node.descendants) == 0:
            layer_node.is_leaf = True
            found_match = False
            for i in range(len(leaves)):
                if leaves[i] == data_node:
                    layer_node.train_index = i
                    found_match = True
            if not found_match:
                print('missing leaf match')
            # create NN submodel, instantiate it to create initial weights
            net = NotTrainableFeedForwardSubModel(self.output_dim, self.num_dense_units, (1, self.num_features))
            net.call(self.dummy_input)
            layer_node.model = net

        # if not a leaf, recursively copy the children
        else:
            for data_child in data_node.descendants:
                new_layer = AdditionLayer(self.layer_shape)
                layer_node.descendants.append(MultifurcatingLayerNode(layer=new_layer, parent=layer_node))
                layers.append(layer_node.descendants[-1])
                self.copy_child_nodes(data_child, layer_node.descendants[-1], layers, leaves)

    def call_child_layers(self, layernode, input_weights, x):
        curr_weights = layernode.layer(input_weights)
        if layernode.height is None or layernode.parent is None or layernode.parent.height is None:
            self.delta_tensor = tf.add(self.delta_tensor, tf.reduce_sum(tf.math.squared_difference(curr_weights, input_weights)))
        else:
            # clamp distance to the configured min distance, prevents dividing by zero, excessive penalties etc.
            mut_distance = tf.math.maximum(tf.abs(tf.subtract(layernode.height, layernode.parent.height)), self.min_distance)
            new_delta = tf.math.divide(
                tf.reduce_sum(tf.abs(tf.subtract(curr_weights, input_weights))), mut_distance)
            self.delta_tensor = tf.add(self.delta_tensor, new_delta)
        # make actual prediction if node is a leaf
        if layernode.is_leaf:
            curr_x = x[layernode.train_index:(layernode.train_index + 1)]
            # layernode.model.set_weights(self.retrieve_nn_weights(curr_weights))
            for i in range(len(self.retrieve_nn_weights(curr_weights))):
                tf.compat.v1.assign(layernode.model.weights[i], self.retrieve_nn_weights(curr_weights)[i])
            return layernode.model(curr_x)
        # otherwise keep recursively passing weights down the tree
        assert len(layernode.descendants) > 0
        if len(layernode.descendants) == 1:
            return tf.identity(self.call_child_layers(layernode.descendants[0], curr_weights, x))
        else:
            return tf.concat([self.call_child_layers(child, curr_weights, x) for child in layernode.descendants],
                             axis=0)

    def call(self, x):
        self.delta_tensor = tf.identity([0.0]) # need to reset the delta to zero at every call
        result = self.call_child_layers(self.root_layer, self.zero_vector, x)
        return result, self.delta_tensor


class EntangledDeltaTreeModel(MultifurcatingTreeModelLinReg):
    """
    Parameter breakdown
    First hidden layer: 3x3 matrix, 3 bias -> 12 total
    Second hidden layer: 1x3 matrix, 1 bias -> 4 total
    Summation: 16 parameters
    """

    def __init__(self, data_tree_root, leaves, layer_shape, min_distance = 1e-7, delta_loss_style='l1', delta_const_1=0.0, delta_const_2=0.001):
        super(EntangledDeltaTreeModel, self).__init__(data_tree_root, leaves, layer_shape, min_distance)
        layers = list()
        assert delta_loss_style in ['l1', 'l1_indicator_approx']
        self.delta_const_1 = delta_const_1
        self.delta_const_2 = delta_const_2
        self.delta_loss_style = delta_loss_style
        self.leaf_weights = tf.Variable([tf.zeros(shape=(3, 3), dtype=tf.float32) for _ in range(len(leaves))], trainable=False)
        self.leaf_biases = tf.Variable([tf.zeros(shape=(3), dtype=tf.float32) for _ in range(len(leaves))], trainable=False)
        self.leaf_activations = tf.Variable([tf.zeros(shape=(1, 3), dtype=tf.float32) for _ in range(len(leaves))], trainable=False)
        self.min_distance = min_distance
        self.layer_shape = layer_shape
        self.addition_layer_0 = AdditionLayer(self.layer_shape)
        self.root_layer = MultifurcatingLayerNode(self.addition_layer_0)
        # replicate the data-tree structure in layers
        self.copy_child_nodes(data_tree_root, self.root_layer, layers, leaves)
        self.zero_vector = tf.zeros(layer_shape)
        self.delta_tensor = tf.identity([0.0])
        # initialize all the layers to be attributes
        for i in range(len(layers)):
            setattr(self, str(i), layers[i].layer)
            layers[i].layer.build(self.layer_shape)

    def calc_delta_loss(self, layernode, curr_weights, prev_weights):
        if self.delta_loss_style == 'l1':
            if layernode.height is None or layernode.parent.height is None:
                self.delta_tensor = tf.add(self.delta_tensor, tf.reduce_sum(tf.abs(tf.subtract(curr_weights, prev_weights))))
            else:
                # clamp distance to the configured min distance, prevents dividing by zero, excessive penalties etc.
                mut_distance = tf.math.maximum(tf.abs(tf.subtract(
                    layernode.height, layernode.parent.height)), self.min_distance)
                new_delta = tf.math.divide(
                    tf.reduce_sum(tf.abs(tf.subtract(curr_weights, prev_weights))), mut_distance)
                self.delta_tensor = tf.add(self.delta_tensor, new_delta)
        """
        new_penalty = (const_1 + x) + log(const_2 + x) - log(const_2)
        """
        if (self.delta_loss_style == 'l1_indicator_approx'):
            x = tf.abs(tf.subtract(prev_weights, curr_weights))
            new_penalty = tf.add(tf.add(x, self.delta_const_1), tf.subtract(tf.math.log(tf.add(x, self.delta_const_2)),
                                                                            tf.math.log(self.delta_const_2)))
            self.delta_tensor = tf.add(self.delta_tensor, tf.reduce_sum(new_penalty))

    def call_child_layers(self, layernode, input_weights, x):
        curr_weights = layernode.layer(input_weights)
        if layernode.parent is not None:
            self.calc_delta_loss(layernode, curr_weights, input_weights)
        # # make actual prediction if node is a leaf
        if layernode.is_leaf:
            curr_x = x[layernode.train_index:(layernode.train_index + 1)]
            # todo: this is the part where we should be taking in an argument, its the only line that changes
            return self.call_nn(curr_x, curr_weights, layernode.train_index)
        # otherwise keep recursively passing weights down the tree
        assert len(layernode.descendants) > 0
        if len(layernode.descendants) == 1:
            return tf.identity(self.call_child_layers(layernode.descendants[0], curr_weights, x))
        else:
            return tf.concat([self.call_child_layers(child, curr_weights, x) for child in layernode.descendants],
                             axis=0)

    def call_nn(self, curr_x, curr_weights, index):
        # lets reshape this bitch
        h1_w = tf.reshape(curr_weights[0:9], (3, 3))
        # update the leaf weights for later analysis
        self.leaf_weights[index].assign(h1_w) # no assignment in eager mode, workaround
        h1_b = curr_weights[9:12]
        self.leaf_biases[index].assign(h1_b)
        out_w = curr_weights[12:15]
        out_b = curr_weights[15]

        # hidden layer 1: matrix multiplication, bias add and activation
        hidden_layer = tf.tanh(tf.add(tf.matmul(curr_x, h1_w), h1_b))
        self.leaf_activations[index].assign(hidden_layer)
        output_layer = tf.add(tf.matmul(hidden_layer, tf.transpose(tf.expand_dims(out_w, axis=0))), out_b)
        return tf.squeeze(output_layer, axis=0)


class RandomEntangledModel(EntangledDeltaTreeModel):
    #  todo: rename layer_shape, it is confusing
    def __init__(self, data_tree_root, leaves, hidden_dims, layer_shape, min_distance = 1e-7, num_activations=7):
        super(RandomEntangledModel, self).__init__(data_tree_root, leaves, layer_shape, min_distance)
        layers = list()
        self.num_activations = num_activations
        self.leaf_weights = tf.Variable([tf.zeros(shape=(layer_shape), dtype=tf.float32) for _ in range(len(leaves))], trainable=False)
        self.leaf_activations = tf.Variable([tf.zeros(shape=(self.num_activations), dtype=tf.float32) for _ in range(len(leaves))], trainable=False)
        self.min_distance = min_distance
        self.hidden_dims = hidden_dims
        self.layer_shape = layer_shape  # a flat layer with the number of weights
        self.addition_layer_0 = AdditionLayer(self.layer_shape)
        self.root_layer = MultifurcatingLayerNode(self.addition_layer_0)
        # replicate the data-tree structure in layers
        self.copy_child_nodes(data_tree_root, self.root_layer, layers, leaves)
        self.zero_vector = tf.zeros(layer_shape)
        self.delta_tensor = tf.identity([0.0])
        # initialize all the layers to be attributes
        for i in range(len(layers)):
            setattr(self, str(i), layers[i].layer)
            layers[i].layer.build(self.layer_shape)

    def call_nn(self, curr_x, curr_weights, index, activation='tanh'):
        self.leaf_weights[index].assign(curr_weights)  # no assignment in eager mode, workaround
        leaf_activations = tf.zeros(shape=1, dtype=tf.float32) # one initial placeholder value, gets removed later

        # reshaping flat weights into the correct shape
        # first layer has shape of curr_dim^2, others are prev_dim * curr_dim
        curr_dim = self.hidden_dims[0]
        prev_dim = self.hidden_dims[0]
        curr_weight_index = 0
        curr_vals = curr_x
        for i in range(1, len(self.hidden_dims)):
            mat_size = prev_dim * curr_dim
            layer_matrix = tf.reshape(curr_weights[curr_weight_index:curr_weight_index+mat_size], (prev_dim, curr_dim))
            curr_weight_index += prev_dim*curr_dim
            layer_bias = curr_weights[curr_weight_index:curr_weight_index+curr_dim]
            curr_weight_index += curr_dim
            curr_vals = tf.tanh(tf.add(tf.matmul(curr_vals, layer_matrix), layer_bias))
            prev_dim = curr_dim
            curr_dim = self.hidden_dims[i]
            leaf_activations = tf.concat((leaf_activations, tf.squeeze(curr_vals)), axis=0)

        # last layer with no activation
        mat_size = prev_dim * curr_dim
        layer_matrix = tf.reshape(curr_weights[curr_weight_index:curr_weight_index + mat_size], (prev_dim, curr_dim))
        curr_weight_index += prev_dim * curr_dim
        layer_bias = curr_weights[curr_weight_index:curr_weight_index + curr_dim]
        output = tf.add(tf.matmul(curr_vals, layer_matrix), layer_bias)
        leaf_activations = tf.concat((leaf_activations, tf.reshape(output, (1,))), axis=0)
        self.leaf_activations[index].assign(leaf_activations[1:])  # no assignment in eager mode, workaround
        return output


class FeedForwardClassificationTree(RandomEntangledModel):
    #  todo: rename layer_shape, it is confusing
    def __init__(self, data_tree_root, leaves, hidden_dims, layer_shape, min_distance = 1e-7):
        super(FeedForwardClassificationTree, self).__init__(data_tree_root, leaves, hidden_dims, layer_shape, min_distance)
        layers = list()
        self.min_distance = min_distance
        self.hidden_dims = hidden_dims
        self.layer_shape = layer_shape  # a flat layer with the number of weights
        self.addition_layer_0 = AdditionLayer(self.layer_shape)
        self.root_layer = MultifurcatingLayerNode(self.addition_layer_0)
        # replicate the data-tree structure in layers
        self.copy_child_nodes(data_tree_root, self.root_layer, layers, leaves)
        self.zero_vector = tf.zeros(layer_shape)
        self.delta_tensor = tf.identity([0.0])
        # initialize all the layers to be attributes
        for i in range(len(layers)):
            setattr(self, str(i), layers[i].layer)
            layers[i].layer.build(self.layer_shape)

    def call_nn(self, curr_x, curr_weights, index, activation='relu'):
        # reshaping flat weights into the correct shape
        # first layer has shape of curr_dim^2, others are prev_dim * curr_dim
        curr_dim = self.hidden_dims[1]
        prev_dim = self.hidden_dims[0]
        curr_weight_index = 0
        curr_vals = curr_x
        for i in range(1, len(self.hidden_dims)):
            mat_size = prev_dim * curr_dim
            layer_matrix = tf.reshape(curr_weights[curr_weight_index:curr_weight_index+mat_size], (prev_dim, curr_dim))
            curr_weight_index += prev_dim*curr_dim
            layer_bias = curr_weights[curr_weight_index:curr_weight_index+curr_dim]
            curr_weight_index += curr_dim
            curr_vals = tf.nn.tanh(tf.add(tf.matmul(curr_vals, layer_matrix), layer_bias))
            prev_dim = curr_dim
            curr_dim = self.hidden_dims[i]

        # last layer
        mat_size = prev_dim * curr_dim
        layer_matrix = tf.reshape(curr_weights[curr_weight_index:curr_weight_index + mat_size], (prev_dim, curr_dim))
        curr_weight_index += prev_dim * curr_dim
        layer_bias = curr_weights[curr_weight_index:curr_weight_index + curr_dim]
        output = tf.add(tf.matmul(curr_vals, layer_matrix), layer_bias)

        return tf.nn.softmax(output)



# # for testing
# import numpy as np
# if __name__ == '__main__':
#     net1 = FeedForwardSubModel(2, 16, [100])
#     net1.call(np.zeros((1, 100), dtype=np.float32))
#     net2 = NotTrainableFeedForwardSubModel(2, 16, [100])
#     print('built the models')



