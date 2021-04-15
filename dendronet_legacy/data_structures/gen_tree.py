import numpy as np

class Node:
    def __init__(self, x, a, is_leaf=False, parent=None):
        self.parent = parent
        self.is_leaf = is_leaf
        self.left = None
        self.right = None
        self.x = x
        self.a = a
        self.y=self.generate_y()
        self.height = None

    def print_tree(self):
        print(self.x)

    def generate_y(self):
        result = 0.0
        for i in range(len(self.a)):
            # result += (self.x * self.a[i]) ** (2*(i+1))  # will always be positive
            result += ((self.x ** i) * self.a[i])  # first term is the bias / intercept term
        return result


class LayerNode:
    def __init__(self, layer, is_leaf=False, parent=None):
        self.layer = layer
        self.is_leaf = is_leaf
        self.parent = parent
        self.left = None
        self.right = None
        self.train_index = None
        self.height = None


class MultifurcatingLayerNode:
    def __init__(self, layer, is_leaf=False, parent=None):
        self.layer = layer
        self.is_leaf = is_leaf
        self.parent = parent
        self.descendants = list()
        self.train_index = None
        self.height = None
        self.model = None


class FungiNode:
    def __init__(self, name=None, is_leaf=False, parent=None, height=1, features=None, target=None):
        self.name = name
        self.is_leaf = is_leaf
        self.parent = parent
        self.height = height
        if features is None:
            self.x = dict()
        else:
            self.x = features
        self.lifestyles = None
        self.y = target
        self.left = None
        self.right = None
        self.is_present = True
        self.placement_vector = list()


def generate_random_tree(depth=10, weights_dim=3, mutation_rate=0.01):
    starting_a = np.random.uniform(size=weights_dim)
    starting_x = np.random.uniform(low=0.0, high=10.0)
    root = Node(x=starting_x, a=starting_a)
    curr_depth = 1
    curr_level = list()
    curr_level.append(root)
    leaves = list()
    # create all internal nodes
    while curr_depth < depth - 1:
        next_level = list()
        curr_depth += 1
        for nd in curr_level:
            # make left child
            left = generate_random_child(nd, is_left=True, a_mutation_rate=mutation_rate)
            # make right child
            right = generate_random_child(nd, is_left=False, a_mutation_rate=mutation_rate)
            next_level.append(left)
            next_level.append(right)
        curr_level = next_level
    # making the leaves
    for nd in curr_level:
        # make left child
        left = generate_random_child(nd, is_left=True)
        # make right child
        right = generate_random_child(nd, is_left=False)
        leaves.append(left)
        leaves.append(right)
    print('Done generating the tree')
    return root, leaves


def generate_random_child(parent_node, is_left, a_mutation_rate=0.01):
    new_x = max(0.0, parent_node.x + np.random.normal())  # mutation, cannot drop below 0
    new_a = parent_node.a.copy()
    for i in range(len(new_a)):
        new_a[i] = max(0.0, new_a[i] + np.random.normal(scale=a_mutation_rate))
    child = Node(x=new_x, a=new_a, parent=parent_node)
    if is_left:
        parent_node.left = child
    else:
        parent_node.right = child
    return child


# running the method for testing
# generate_random_tree()
