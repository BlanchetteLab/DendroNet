import torch
import torch.nn as nn
from models.pytorch_models import DendroFCNN
from data_structures.entangled_data_simulation import BernoulliTree


EPOCHS = 1000
lr = 0.001
dpf = 0.01

if __name__ == '__main__':

    data_tree = BernoulliTree(mutation_rate=0.0, depth=11, num_leaves=2, low=0.0, high=1.0,
                              mutation_prob=0.0, mutation_style='exponential')
    x = list()
    y = list()
    for leaf in data_tree.leaves:
        x.append(leaf.x)
        y.append(leaf.y)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    model = DendroFCNN(data_root=data_tree.tree)
    leaves = model.leaf_list
    print('Tree contains ' + str(len(leaves)) + ' leaves')
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(EPOCHS):
        print('Epoch: ' + str(epoch))
        optimizer.zero_grad()
        preds = torch.cat([model(leaves[i], x[i]).reshape(1, 1) for i in range(len(leaves))])
        delta_loss = model.calc_delta_loss()
        loss = loss_fn(preds, y) + (dpf * delta_loss)
        loss.backward()
        optimizer.step()
        print('Total loss: ' + str(loss.item()))
        print('Delta loss: ' + str(dpf * delta_loss))


print('done')
