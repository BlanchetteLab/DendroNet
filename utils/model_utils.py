import torch
import numpy as np


def split_indices(idx_list, seed=0, train_percentage=0.7):
    """
    param idx_list: a list of indices (or IDs) to be split into train and test sets
    param seed: random seed for repeatability
    train_percentage: portion of the data allocated to train set
    returns: set of train indices and set of test indices
    """
    np.random.seed(seed=seed)
    train_idx = list()
    test_idx = list()
    for idx in idx_list:
        if np.random.uniform(0.0, 1.0) < train_percentage:
            train_idx.append(idx)
        else:
            test_idx.append(idx)
    return train_idx, test_idx


"""
-This is for use on trees, use build_parent_path_mat_dag if the data is a non-tree DAG
"""
def build_parent_path_mat(parent_child_mat, num_edges=None):
    """
    param parent_child_mat: np binary array, rows-> parent cols-> child, first row must be root
    -note that the parent-child mat must be topologically ordered
    return: parent_path matrix: np array, rows->edges cols->nodes
    """
    num_nodes = parent_child_mat.shape[0]
    # if num_edges is not passed in, counting the number of edges above the diagonal
    if num_edges is None:
        num_edges = np.sum(np.triu(parent_child_mat, 1))

    parent_path = np.zeros(shape=(num_nodes, num_edges), dtype=np.float32)
    edge_index = 0

    for node_index in range(1, num_nodes):  # skipping the root node, which we know has an empty parent path
        # edge to parent becomes a new edge
        parent_path[node_index, edge_index] = 1.0
        edge_index += 1
        # find the parent node via edge mat, add parent path values
        parent_node_idx = np.where(parent_child_mat[:, node_index] == 1)[0]
        prev_pp_idx = np.where(parent_path[parent_node_idx] == 1.0)
        for idx in prev_pp_idx:
            parent_path[node_index, idx] = 1.0

    # taking the transpose so that every column holds all the relevant edges for a node
    return np.transpose(parent_path)


def build_parent_path_mat_dag(parent_child_mat, num_edges=None):
    """
    param parent_child_mat: np binary array, rows-> parent cols-> child
    -note that the parent-child mat must be topologically ordered
    param num_edges: can be passed in directly, or left as None and counted in the upper diagonal
    return: parent_path matrix: np array, rows->edges cols->nodes
    """
    num_nodes = parent_child_mat.shape[0]
    # if num_edges is not passed in, counting the number of edges above the diagonal
    if num_edges is None:
        num_edges = np.sum(np.triu(parent_child_mat, 1))

    # add the edges that will go from each root to the root weights
    root_edges = 0
    for node_idx in range(num_nodes):
        parent_edges = np.where(parent_child_mat[:, node_idx] == 1)[0]
        valid_parents = [p_e for p_e in parent_edges if p_e < node_idx]
        if len(valid_parents) == 0:
            root_edges += 1
    num_edges += root_edges

    parent_path = np.zeros(shape=(num_nodes, num_edges), dtype=np.float32)
    edge_index = 0
    edges_broken = 0

    """
    This is now different in that there is a delta associated with the 'edge'
    between the root_weights vector and every root in the DAG
    """
    for node_index in range(0, num_nodes):
        # count the number of valid parents, discarding those later in topological order (breaking cycles)
        parent_edges = np.where(parent_child_mat[:, node_index] == 1)[0]
        valid_parents = [p_e for p_e in parent_edges if p_e < node_index]
        edges_broken += len(parent_edges) - len(valid_parents)
        # if the node is a topological root, give it an edge entry to the root weights
        if len(valid_parents) == 0:
            parent_path[node_index, edge_index] = 1.0
            edge_index += 1
        # otherwise, it gets an edge entry for each parent, plus all the parent node edge entries
        else:
            for parent_idx in valid_parents:
                parent_path[node_index, edge_index] = 1.0
                edge_index += 1
                prev_pp_idx = np.where(parent_path[parent_idx] > 0.0)  # values are weighted and positive
                for pp_entry in prev_pp_idx:
                    parent_path[node_index, pp_entry] += parent_path[parent_idx, pp_entry]
            # weight all the entries in the current parent_path by the number of parents
            for i in range(num_nodes):
                parent_path[node_index, i] /= len(valid_parents)

    print(str(edges_broken) + ' edges pruned during parent path construction')
    # taking the transpose so that every column holds all the relevant edges for a node
    return np.transpose(parent_path)

"""
Takes in list of relevant indices, 
for use with the dataloader class for batching when full dataset is an array in memory
"""
class IndicesDataset(torch.utils.data.Dataset):
  def __init__(self, sample_indices):
        self.sample_indices = sample_indices

  def __len__(self):
        return len(self.sample_indices)

  def __getitem__(self, index):
        'Generates an index for one sample of data'
        # Select sample
        sample_idx = self.sample_indices[index]

        return sample_idx