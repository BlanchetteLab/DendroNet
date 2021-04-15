import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from models.dendronet_models import DendroMatrixLinReg
from simulated_data_applications.generate_graph import gen_random_grid
from utils.model_utils import build_parent_path_mat_dag, split_indices, IndicesDataset

"""
An example application, suitable as a template / starting point for those looking to use DendroNet on their own data
"""

# flag to use cuda gpu if available
USE_CUDA = True
print('Using CUDA: ' + str(USE_CUDA))
device = torch.device("cuda:0" if torch.cuda.is_available() and USE_CUDA else "cpu")

# some other hyper-parameters for training
LR = 0.001
BATCH_SIZE = 8
EPOCHS = 1000
DPF = 0.1

"""
Some preprocessing must be done by the user on the dataset of interest to generate 3 components:
1. A parent-child matrix, where an entry of 1 indicates an edge between a parent node (row) and child node (column)
2. A matrix X containing all of the node features, with the same row-order as the parent-child matrix
3. An array y containing all of the target y-values, in the same order as the feature matrix rows

In the case that not all nodes in the graph have features (for example, if the data is a phylogenetic tree and 
only leaves have data available), some accounting must be done to track the mapping between parent-child rows 
and corresponding examples in X and y. See tree_tutorial.py for an example.

In this case, we will generate these 3 components from a simulated grid, where each node is the parent of its 
neighbours to the left and to the bottom, and the function defining the relationship between the node features and 
target y-values is more similar in nodes that are close together in the grid than nodes that are far apart. 
"""
parent_child, X, y = gen_random_grid(size=5)
num_features = len(X[0])
num_nodes = len(parent_child[0])

"""
DendroNet uses 3 matrices when performing calculations, in addition to the X and y matrices.
1. The parent-path matrix, which can be easily obtained from the parent-child matrix
2. A matrix holding the weights applied to the model at the root of the graph 
4. A delta matrix, which will hold the mutations applied to each of the root weights along the graph edges
"""

"""
-generating the parent-path matrix from the parent-child matrix
-if many / all edges are getting pruned, check that the parent/child relationship wasn't inversed by taking
the transpose of the parent-child matrix
"""
parent_path_tensor = build_parent_path_mat_dag(parent_child)
num_edges = len(parent_path_tensor)
# tensor with the starting weights at the root node, assumes bias feature is included
root_weights = np.zeros(shape=num_features)

# each column holds a trainable delta vector for each edge in the graph
edge_tensor_matrix = np.zeros(shape=(num_features, num_edges))

"""
Now we have all the components, and can create an instance of the DendroNet model specific to our graph architecture
We will use a linear regressor as the base architecture
"""
dendronet = DendroMatrixLinReg(device, root_weights, parent_path_tensor, edge_tensor_matrix)

"""
We typically want to split into a train and test/validation set. We will do this by assigning indices to either group so 
that we can keep the mapping between parent_path column ID and the corresponding sample in X and y

We can then use the IndicesDataset class to perform shuffle-batching during training
"""

train_idx, test_idx = split_indices(range(len(y)))

# creating idx dataset objects for batching
train_set = IndicesDataset(train_idx)
test_set = IndicesDataset(test_idx)

# Setting some parameters for shuffle batch
params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 0}

train_batch_gen = torch.utils.data.DataLoader(train_set, **params)
test_batch_gen = torch.utils.data.DataLoader(test_set, **params)

# converting X and y to tensors, and transferring to GPU if the cuda flag is set
X = torch.tensor(X, dtype=torch.double, device=device)
y = torch.tensor(y, dtype=torch.double, device=device)

# creating the loss function and optimizer
loss_function = nn.MSELoss()
if torch.cuda.is_available() and USE_CUDA:
    loss_function = loss_function.cuda()
optimizer = torch.optim.SGD(dendronet.parameters(), lr=LR)

# running the training loop
for epoch in range(EPOCHS):
    print('Train epoch ' + str(epoch))
    # we'll track the running loss over each batch so we can compute the average per epoch
    running_loss = 0.0
    # getting a batch of indices
    for step, idx_batch in enumerate(tqdm(train_batch_gen)):
        optimizer.zero_grad()
        # dendronet takes in a set of examples from X, and the corresponding column indices in the parent_path matrix
        y_hat = dendronet.forward(X[idx_batch], idx_batch)
        # collecting the two loss terms
        delta_loss = dendronet.delta_loss()
        # idx_batch is also used to fetch the appropriate entries from y
        train_loss = loss_function(y_hat, y[idx_batch])
        running_loss += float(train_loss.detach().cpu().numpy())
        loss = train_loss + (delta_loss * DPF)
        loss.backward(retain_graph=True)
        optimizer.step()
    print('Average MSE loss: ', str(running_loss / step))

# With training complete, we'll run the test set. We could use batching here as well if the test set was large
with torch.no_grad():
    y_hat = dendronet.forward(X[test_idx], test_idx)
    loss = loss_function(y_hat, y[test_idx])
    delta_loss = dendronet.delta_loss()
    print('Final Delta loss:', float(delta_loss.detach().cpu().numpy()))
    print('Test set MSE:', float(loss.detach().cpu().numpy()))
