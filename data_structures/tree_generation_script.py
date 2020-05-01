from utils import dump_dict
from data_structures.entangled_data_simulation import EntangledTree, store_tree_and_leaves

seeds = [0, 1, 2, 3, 4]
base_folder = 'tree_storage/seed_'

config = {
    'depth': 9,
    'mutation_rate': 0.1,
    'num_leaves': 1,
    'low': 0,
    'high': 5,
}

print('Generating trees with seeds ' + str(seeds))
print('Using config ' + str(config))
for seed in seeds:
    data_tree = EntangledTree(seed=seed, depth=config['depth'], mutation_rate=config['mutation_rate'],
                              num_leaves=config['num_leaves'], low=config['low'], high=config['high'])
    folder_name = base_folder + str(seed)
    store_tree_and_leaves(data_tree.tree, data_tree.leaves, folder_name)
    dump_dict(config, folder_name)
    print('Stored a tree in folder ' + folder_name)
print('All trees created and stored')
