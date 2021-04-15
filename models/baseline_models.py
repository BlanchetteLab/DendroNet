import torch
import torch.nn as nn

"""
Single output version, for use w/ BCELoss, use_bias=False if bias is already a feature
"""
class LogRegModel(nn.Module):
    def __init__(self, input_dim, use_bias=False):
        super(LogRegModel, self).__init__()
        self.lin_1 = nn.Linear(input_dim, 1, bias=use_bias)

    def forward(self, x):
        return torch.sigmoid(self.lin_1(x)).squeeze()


class LinRegModel(nn.Module):
    def __init__(self, input_dim, use_bias=False):
        super(LinRegModel, self).__init__()
        self.lin_1 = nn.Linear(input_dim, 1, bias=use_bias)

    def forward(self, x):
        return self.lin_1(x).squeeze()


"""
Baseline for use with NNDendroMatrix
"""
class NeuralNet2Layer(nn.Module):
    def __init__(self, num_features, layer_sizes, use_bias=False):
        """
        layer_sizes: number of units for each of the two layers (i.e. output size of each layer)
        use_bias: set to False if bias is already a feature
        """
        super(NeuralNet2Layer, self).__init__()
        assert len(layer_sizes) == 2, 'unsupported number of layer sizes'
        self.layer_1 = nn.Linear(num_features, layer_sizes[0], bias=use_bias)
        self.layer_2 = nn.Linear(layer_sizes[0], layer_sizes[1], bias=use_bias)

    # returning raw scores for use with CrossEntropyLoss or MSE, assumes bias is already in
    def forward(self, x):
        l1_out = torch.nn.functional.relu(self.layer_1(x))
        return self.layer_2(l1_out)