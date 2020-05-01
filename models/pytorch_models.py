import torch
import torch.nn as nn
import torch.nn.functional as F


class DendroLinNode:
    def __init__(self, weight, parent=None, is_leaf=False, leaf_index=0):
        self.weight = weight
        self.children = list()
        self.is_leaf = is_leaf
        self.parent = parent
        self.leaf_index = leaf_index
        self.weight_list = list()
        self.weight_tensor = None


class LinRegModel(nn.Module):
    def __init__(self, input_dim, output_dim=1, use_bias=True):
        super(LinRegModel, self).__init__()
        self.lin_1 = nn.Linear(input_dim, output_dim, bias=use_bias)

    def forward(self, x):
        return self.lin_1(x)


# assumption is that bias has been added as an additional feature
class LogRegModel(nn.Module):
    def __init__(self, input_dim, use_bias=False):
        super(LogRegModel, self).__init__()
        self.lin_1 = nn.Linear(input_dim, 2, bias=use_bias)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        return self.softmax(self.lin_1(x))


class SimpleFCNN(nn.Module):
    # print('change back to tanh')
    def __init__(self, input_dim, num_hidden=3, output_dim=1):
        super(SimpleFCNN, self).__init__()
        self.lin_1 = nn.Linear(input_dim, num_hidden)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.out_layer = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        return self.out_layer(self.tanh(self.lin_1(x)))


class LinearVAE(nn.Module):
    def __init__(self, feature_dim, encoding_dim):
        super(LinearVAE, self).__init__()
        self.encoder = nn.Linear(feature_dim, encoding_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(encoding_dim, feature_dim)

    def forward(self, x):
        encoding = self.relu(self.encoder(x))
        decoding = self.decoder(encoding)
        return encoding, decoding

"""
Discovery note: with initializing the weights all the zero, they end up being driven to the
exact same values, unless weight decay is used. By starting them at random values, they converge to separate end 
points
"""

"""
Assumes that features include a dummy bias feature
"""
class DendroLinReg(nn.Module):
    def __init__(self, input_dim, data_root, output_dim=1, use_cuda=False, device=None):
        super(DendroLinReg, self).__init__()
        self.use_cuda = use_cuda
        self.device = device
        self.input_dim = input_dim
        self.data_root = data_root
        self.output_dim = output_dim
        self.leaf_list = list()
        self.delta_tensor = torch.tensor([0.0], requires_grad=True)
        self.layer_counter = 0
        if use_cuda:
            self.delta_tensor.to(device)
        self.initial_weights = torch.empty(size=(1, self.input_dim))
        torch.nn.init.normal_(self.initial_weights, std=0.5)
        self.initial_weights = nn.Parameter(self.initial_weights, requires_grad=True)
        if use_cuda:
            self.initial_weights.to(self.device)
        self.model_root = DendroLinNode(self.initial_weights)
        self.construct_model_tree(self.model_root, self.data_root)
        # self.finalize_leaves()

    def construct_model_tree(self, model_node, data_node):
        for data_child in data_node.descendants:
            # child_weight = nn.Parameter(torch.zeros(size=(1, self.input_dim)), requires_grad=True)
            if data_child.is_leaf:
                child_weight = None
            else:
                child_weight = torch.empty(size=(1, self.input_dim))
                torch.nn.init.normal_(child_weight, std=0.0001)
                child_weight = nn.Parameter(child_weight, requires_grad=True)
                setattr(self, str('layer_' + str(self.layer_counter)), child_weight)
                self.layer_counter += 1
            if self.use_cuda:
                child_weight.to(self.device)
            child_node = DendroLinNode(weight=child_weight, is_leaf=data_child.is_leaf,
                                       parent=model_node)
            model_node.children.append(child_node)
            self.construct_model_tree(child_node, data_child)
        """
        concatenating the relevant parent tensors, allowing them to be accessed in forward
        NOTE: we now do not allow trainable weights connecting to leaf nodes, as we work with many examples
        per leaf at the last layer of the tree
        """
        if not model_node.is_leaf:
            model_node.weight_list.append(model_node.weight)
        curr_parent = model_node.parent
        while curr_parent is not None:
            model_node.weight_list.append(curr_parent.weight)
            curr_parent = curr_parent.parent
        if model_node.is_leaf:
            self.leaf_list.append(model_node)

    """
    New pattern: given a leaf, repeatedly add the layers for all parents. Use the functional linear function
    Later try and switch forward to taking a list of leaves.
    """
    #
    # def finalize_leaves(self):
    #     for leaf in self.leaf_list:
    #         leaf.weight_tensor = torch.cat(leaf.weight_list)
    #         if self.use_cuda:
    #             leaf.weight_tensor.to(self.device)

    def calc_delta_loss(self):
        with torch.no_grad():
            self.delta_tensor = torch.tensor([0.0])
            self.delta_tensor.to(self.device)
        self.recursively_add_deltas(self.model_root)
        return self.delta_tensor

    def recursively_add_deltas(self, curr_node):
        for child_node in curr_node.children:
            if not child_node.is_leaf:
                self.delta_tensor += torch.sum(torch.abs(child_node.weight))
                self.recursively_add_deltas(child_node)

    def forward(self, leaf_node, x):
        return F.linear(x, torch.sum(torch.cat(leaf_node.weight_list), dim=0))
        # return F.linear(x, torch.sum(leaf_node.weight_tensor, dim=0))


#assuming that a bias feature is added to the data already
class DendroLogReg(DendroLinReg):
    def __init__(self, data_root, num_features, output_dim=2):
        self.weight_dim = (num_features * 2)
        self.num_features = num_features
        super(DendroLogReg, self).__init__(input_dim=self.weight_dim, data_root=data_root, output_dim=output_dim)

    def forward(self, leaf_node, x):
        param_vector = torch.sum(torch.cat(leaf_node.weight_list), dim=0)
        fc_w = param_vector.view(2, self.num_features)
        return F.softmax(F.linear(x, fc_w))
        # return F.softmax(F.linear(x, torch.transpose(fc_w, 0, -1)))


"""
implementing this exactly as in the pytorch disentanglement experiment:
3x3 layer and 3x1 bias
3x1 layer and 1x1 bias
Total of 16 parameters
"""
class DendroFCNN(DendroLinReg):
    def __init__(self, data_root, weight_dim=16, output_dim=1, use_cuda=False, device=None):
        """
        :param weight_dim: gets passed to parent, it will result in the weights having shape (1, weight_dim)
        and needing to be reshaped appropriately at the leaves inside the forward method
        """
        super(DendroFCNN, self).__init__(input_dim=weight_dim, data_root=data_root, output_dim=output_dim,
                                         use_cuda=use_cuda, device=device)

    def forward(self, leaf_node, x):
        weight_vector = torch.sum(torch.cat(leaf_node.weight_list), dim=0)
        # weight_vector = torch.sum(leaf_node.weight_tensor, dim=0)
        fc1_w = weight_vector[0:9].view(3, 3)
        fc1_b = weight_vector[9:12]
        fc2_w = weight_vector[12:15]
        fc2_b = weight_vector[15]

        fc1_out = F.tanh(F.linear(x, fc1_w, bias=fc1_b))
        # fc1_out = F.relu(F.linear(x, fc1_w, bias=fc1_b))
        return F.linear(fc1_out, fc2_w, bias=fc2_b)

    @staticmethod
    def return_weights(leaf_node):
        with torch.no_grad():
            # pretty well mimics the forward call
            weight_vector = torch.sum(torch.cat(leaf_node.weight_list), dim=0)
            return {
                'fc1_w': weight_vector[0:9].view(3, 3).detach().numpy(),
                'fc1_b': weight_vector[9:12].detach().numpy(),
                'fc2_w': weight_vector[12:15].detach().numpy(),
                'fc2_b': weight_vector[15].detach().numpy()
            }


class DendroVAE(DendroLinReg):
    def __init__(self, data_root, weight_dim, feature_dim, encoding_size):
        super(DendroVAE).__init__(input_dim=weight_dim, data_root=data_root)
        self.feature_dim = feature_dim
        self.encoding_size = encoding_size
        self.mat_size = feature_dim * encoding_size

    def forward(self, leaf_node, x):
        weight_vector = torch.sum(torch.cat(leaf_node.weight_list), dim=0)
        # encoding layer
        weight_idx = 0
        fc1_w = weight_vector[weight_idx:self.mat_size].view(self.encoding_size, self.feature_dim)
        weight_idx += self.mat_size
        fc1_b = weight_vector[weight_idx:weight_idx+self.encoding_size]
        weight_idx += self.encoding_size
        encoding = F.linear(x, fc1_w, bias=fc1_b)
        encoding = F.relu(encoding)

        # decoding layer
        fc2_w = weight_vector[weight_idx:weight_idx+self.mat_size].view(self.feature_dim, self.encoding_size)
        weight_idx += self.mat_size
        fc2_b = weight_vector[weight_idx:weight_idx+self.feature_dim]
        decoding = F.linear(encoding, fc2_w, bias=fc2_b)

        return encoding, decoding
