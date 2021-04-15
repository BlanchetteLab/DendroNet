import json, os
import jsonpickle


class PatricNode(object):
    def __init__(self, name=None, is_leaf=False, parent=None, height=1.0, features=None, target=None, level=None):
        self.name = name
        self.is_leaf = is_leaf
        self.parent = parent
        self.height = height
        if features is None:
            self.x = list()
        else:
            self.x = features
        self.y = target
        self.descendants = list()
        self.is_present = True
        self.level = level
        self.bias_feature = list()
        self.taxonomy_dict = dict()
        self.expanded_features = list()


def store_tree_and_leaves(tree, leaves, folder_name, tree_basename='tree.json', leaf_basename='leaf.json'):
    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, tree_basename), 'w') as out:
        json.dump(jsonpickle.encode(tree), out, indent=4)
    out.close()
    leaf_dict = {'leaves': [leaf.name for leaf in leaves]}
    with open(os.path.join(folder_name, leaf_basename), 'w') as out:
        json.dump(leaf_dict, out, indent=4)
    out.close()


def load_tree_and_leaves(folder_name, tree_basename='tree.json', leaf_basename='leaf.json'):
    with open(os.path.join(folder_name, tree_basename)) as file:
        js_string = json.load(file)
    file.close()
    tree = jsonpickle.decode(js_string)
    with open(os.path.join(folder_name, leaf_basename)) as file:
        leaf_dict = json.load(file)
    file.close()
    leaf_ids = leaf_dict['leaves']
    leaves = [PatricNode() for _ in range(len(leaf_ids))]
    recursively_recover_leaves(tree, leaf_ids, leaves)
    return tree, leaves


def recursively_recover_leaves(node, leaf_ids, leaf_list):
    if node.is_leaf:
        # store the PatricNode at the correct index
        assert node.name in leaf_ids
        leaf_list[leaf_ids.index(node.name)] = node
    else:
        for child in node.descendants:
            recursively_recover_leaves(child, leaf_ids, leaf_list)


# remove after testing
# collecting the leaves from the preprocessed file
# if __name__ == "__main__":
#     test_leaves = list()
#     with open(os.path.join('patric_application', 'multiclass_clostridium_samples.csv')) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         line_count = 0
#         for row in csv_reader:
#             if line_count == 0:
#                 print(f'Column names are {", ".join(row)}')
#             else:
#                 pass
#         line_count += 1
#
#     parse_tree(filepath=os.path.join('patric_application', 'PATRIC_phylogeny_tree_clostridiales.nwk'), leaves=[])
