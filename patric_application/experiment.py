import os
import argparse
import pandas as pd
from patric_application.process_genome_lineage import load_tree_and_leaves


"""
For Georgi: 
-The only relevant parameters for now are the last two (tree-path, label-file), which will point to the outputs from 
the preprocessor files
-Don't worry about the other parameters yet, we will use them once we are training DendroNet
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, metavar='N')
    parser.add_argument('--early-stopping', type=int, default=3, metavar='E',
                        help='Number of epochs without improvement before early stopping')
    parser.add_argument('--seed', type=int, default=[0], metavar='S',
                        help='random seed for train/valid split (default: 0)')
    parser.add_argument('--validation-interval', type=int, default=1, metavar='VI')
    parser.add_argument('--dpf', type=float, default=0.1, metavar='D',
                        help='scaling factor applied to delta term in the loss (default: 1.0)')
    parser.add_argument('--l1', type=float, default=1.0)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='patric', metavar='O')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
    parser.add_argument('--runtest', dest='runtest', action='store_true')
    parser.add_argument('--no-runtest', dest='runtest', action='store_false')
    parser.set_defaults(runtest=False)
    parser.add_argument('--tree-path', type=str, default=os.path.join('data_files', 'patric_tree_storage', 'betalactam')
                        , help='folder to look in for a stored tree structure')
    parser.add_argument('--label-file', type=str, default=os.path.join('data_files', 'betalactam_firmicutes_samples.csv'),
                        metavar='LF', help='file to look in for labels')
    args = parser.parse_args()

    data_tree, leaves = load_tree_and_leaves(args.tree_path)
    # annotating leaves with labels and features
    labels_df = pd.read_csv(args.label_file, dtype=str)

    """
    Georgi:
    labels_df contains data about each species: it's ID, phenotype (y), and features (x) 
    The data is in an annoying format, here's a loop showing how to access it row-by-row
    and match it to a leaf in the 'leaves' object 
    """
    for row in labels_df.itertuples():
        for leaf in leaves:
            if leaf.name == getattr(row, 'ID'):  # we have matched a leaf to it's row in labels_df
                phenotype = eval(getattr(row, 'Phenotype'))[0]  # the y value
                features = eval(getattr(row, 'Features'))  # the x value

    """
    For Georgi - here is where you can start the programming task we discussed. There are 3 components we need:
    1. A matrix X, where each row contains the features for a bacteria
    2. A vector y, where each entry contains the phenotype for a bacteria. This should be in the same order as X; i.e.
    entry 0 in y is the phenotype for row 0 in X
    3. A parent-child matrix for the tree that is defined by the  structure 'data_tree', with some more details below:
        -This parent-child matrix has rows for all nodes in data_tree (including the internal ones)
        
        -Each row corresponds to a parent node, and each column to a child node, i.e. a 1 is entered at 
        position (row 1, col 2) if the node corresponding to row 1 is the parent of the node corresponding to column 2
        
        -The rows should be in descending order; i.e. row/col 0 is the root, row/col 1 and 2 are the first layer below the root
        
        -For each row, we need a mapping which tells us the appropriate entry in X that stores info for the relevant 
        species. This could be a list of tuples, i.e. (parent-child-row-index, entry-in-X-index). I would suggest using 
        the ID field to create this list of tuples as you are filling in the parent-child matrix
    """

    print('Data loaded')