import numpy as np

####################################
### CALCULATING ENTROPY AND GAIN ###
####################################
def calculate_entropy(y):
    counts = np.unique(y, return_counts=True)[1]
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_gain_numerical(X, y, attribute):
    prev_entropy = calculate_entropy(y)

    sorted_vals = np.argsort(X[:, attribute])
    sorted_x = X[sorted_vals, attribute]
    sorted_y = y[sorted_vals]

    best_gain = 0
    best_threshold = None

    for i in range(1, len(sorted_y)):
        # if the labels are the same or the values are the same, skip
        if sorted_y[i] == sorted_y[i-1] or sorted_x[i] == sorted_x[i-1]:
            continue
        
        threshold = (sorted_x[i] + sorted_x[i-1]) / 2
        
        left_y = sorted_y[:i]
        right_y = sorted_y[i:]

        weight_left = len(left_y) / len(sorted_y)
        weight_right = len(right_y) / len(sorted_y)

        weighted_entropy = weight_left * calculate_entropy(left_y) + weight_right * calculate_entropy(right_y)
        gain = prev_entropy - weighted_entropy

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_gain, best_threshold

def calculate_gain_categorical(X, y, attribute):
    prev_entropy = calculate_entropy(y)

    vals, counts = np.unique(X[:, attribute], return_counts=True)
    weighted_entropy = 0

    for i in range(len(vals)):
        subset_y = y[X[:, attribute] == vals[i]]
        weighted_entropy += (counts[i] / len(y)) * calculate_entropy(subset_y)

    return prev_entropy - weighted_entropy

##########################
### GENERAL TREE CLASS ###
##########################
class Node:
    def __init__(self, attribute=None, children=None, leaf_result=None, threshold=None, is_numerical=False):
        self.attribute = attribute
        self.children = children
        self.leaf_result = leaf_result # only for leaf nodes
        self.threshold = threshold # only for numerical nodes
        self.is_numerical = is_numerical # only for numerical nodes

######################################
### CONSTRUCTING THE DECISION TREE ###
######################################
def build_tree(X, y, attributes, depth, max_depth, min_gain, min_size_for_split):
    # if the number of training examples is less than minimum threshold
    if len(y) < min_size_for_split:
        vals, counts = np.unique(y, return_counts=True)
        return Node(leaf_result=vals[np.argmax(counts)])

    # if we've reached the maximum allowed depth
    if depth >= max_depth:
        vals, counts = np.unique(y, return_counts=True)
        return Node(leaf_result=vals[np.argmax(counts)])

    # if all remaining labels are exactly the same
    if len(np.unique(y)) == 1:
        return Node(leaf_result=y[0])

    # if there is no more attributes, take the most common label (leaf node)
    if len(attributes) == 0:
        vals, counts = np.unique(y, return_counts=True)
        most_common_label = vals[np.argmax(counts)]
        return Node(None, None, most_common_label)
    
    # the node is not a leaf node so find the best attribute
    m = int(np.sqrt(len(attributes)))
    selected_attributes = np.random.choice(attributes, m, replace=False)

    best_attribute = None
    best_gain = -1
    best_threshold = None
    is_best_numeric = False

    for attribute in selected_attributes:
        # if we are dealing with a numeric split
        if not isinstance(X[0, attribute], str):
            gain, threshold = calculate_gain_numerical(X, y, attribute)
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute
                best_threshold = threshold
                is_best_numeric = True
        # if we are dealing with a categorical split
        else:
            gain = calculate_gain_categorical(X, y, attribute)
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute
                is_best_numeric = False

    # additional stopping criterion, check if gain isn't good enough
    if best_gain <= min_gain:
        vals, counts = np.unique(y, return_counts=True)
        return Node(leaf_result=vals[np.argmax(counts)])

    if is_best_numeric:
        # split based on threshold
        left_tree = X[:, best_attribute] <= best_threshold
        right_tree = X[:, best_attribute] > best_threshold
        
        left_child = build_tree(X[left_tree], y[left_tree], attributes, depth + 1, max_depth, min_gain, min_size_for_split)
        right_child = build_tree(X[right_tree], y[right_tree], attributes, depth + 1, max_depth, min_gain, min_size_for_split)
        
        return Node(attribute=best_attribute, children={"left": left_child, "right": right_child}, 
                    threshold=best_threshold, is_numerical=True)
    else:
        # split based on unique categories
        children = {}
        child_paths = np.unique(X[:, best_attribute])
        for path in child_paths:
            mask = X[:, best_attribute] == path
            children[path] = build_tree(X[mask], y[mask], attributes, depth + 1, max_depth, min_gain, min_size_for_split)
        
        return Node(attribute=best_attribute, children=children, is_numerical=False)


##########################
### TRAVERSING THE TREE ###
##########################
def traverse(root, sample):
    # if we have hit a leaf
    if root.leaf_result is not None:
        return root.leaf_result

    # traversing down the tree
    attribute_value = sample[root.attribute]
    if root.is_numerical:
        if attribute_value <= root.threshold:
            return traverse(root.children["left"], sample)
        else:
            return traverse(root.children["right"], sample)
    else:
        # if we never saw category in training
        if attribute_value not in root.children:
            return None
        child = root.children[attribute_value]
        return traverse(child, sample)