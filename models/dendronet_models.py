import torch
import torch.nn as nn


class DendroMatrixLinReg(nn.Module):
    def __init__(self, device, root_weights, path_mat, delta_mat, p=1, init_deltas=False):
        """
        param p: type of norm to take for dendronet loss
        """
        super(DendroMatrixLinReg, self).__init__()
        self.device = device
        self.path_mat = torch.tensor(path_mat, device=device, dtype=torch.double)

        self.p = p
        self.root_weights = nn.Parameter(torch.tensor(root_weights, device=device, dtype=torch.double, requires_grad=True))
        torch.nn.init.normal_(self.root_weights, mean=0.0, std=0.01)
        self.delta_mat = nn.Parameter(torch.tensor(delta_mat, device=device, dtype=torch.double, requires_grad=True))
        if init_deltas:
            torch.nn.init.normal_(self.delta_mat, mean=0.0, std=0.01)

    def delta_loss(self):
        return torch.norm(self.delta_mat, p=self.p)

    # node_idx identifies the paths relevant to all samples in x, in the same order
    def forward(self, x, node_idx):
        effective_weights = torch.add(self.root_weights, torch.matmul(self.delta_mat, self.path_mat[:, node_idx]).T)
        # this works for linreg with bias-in only
        return torch.sum((x * effective_weights), dim=1)


class DendroMatrixLogReg(DendroMatrixLinReg):
    def __init__(self, device, root_weights, path_mat, delta_mat, p=1, init_deltas=False):
        super(DendroMatrixLogReg, self).__init__(device, root_weights, path_mat, delta_mat, p, init_deltas)

    # node_idx identifies the paths relevant to all samples in x, in the same order
    def forward(self, x, node_idx):
        effective_weights = torch.add(self.root_weights, torch.matmul(self.delta_mat, self.path_mat[:, node_idx]).T)
        # this works for logreg with bias-in only, note that this is a one-output version requiring BCELoss
        return torch.sigmoid(torch.sum((x * effective_weights), dim=1))

    # todo: incorporate into forward to eliminate repeat code? Currently used for feature importance analysis
    def get_effective_weights(self, node_idx):
        return torch.add(self.root_weights, torch.matmul(self.delta_mat, self.path_mat[:, node_idx]).T)