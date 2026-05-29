import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TreeNode:
    def __init__(self, attribute=None, is_leaf=False, prediction=None, threshold=None, is_numeric=0):
        self.attribute = attribute    # attribute to split on (None for leaf)
        self.is_leaf = is_leaf        # whether this node is a leaf
        self.prediction = prediction  # class prediction if leaf 
        self.threshold = threshold    # threshold for numerical split
        self.is_numeric = is_numeric  # 1 for numerical split, 0 for categorical split
        self.children = {}            # stores children nodes if not leaf

class DecisionTree:
    def __init__(self, criterion="information_gain", heuristic=False, max_depth=None):
        self.criterion = criterion
        self.heuristic = heuristic
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        # store categories for categorical features 
        self.all_attr_val = {col: X[col].unique() for col in X.columns if not self.is_numeric_attribute(X, col)}
        attributes = list(X.columns)
        self.root = self.build_tree(X, y, attributes, depth=0)

    def is_numeric_attribute(self, X, attribute):
        # check if the attribute is numeric by checking the dtype of the column in the df
        return pd.api.types.is_numeric_dtype(X[attribute])

    def class_probabilities(self, y):
        # returns the array with probabilities of the labels in y
        _ , counts = np.unique(y, return_counts=True) 
        probabilities = counts / np.sum(counts) 
        return probabilities

    def entropy(self, y):
        # calculates the entropy of the labels in y
        class_prob = self.class_probabilities(y)
        return -np.sum(class_prob * np.log2(class_prob))

    def information_gain_numeric(self, X, y, attribute):
        unique_values = np.sort(X[attribute].unique())
        
        # calculate the thresholds, the midpoints between consecutive unique values
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        orig_entropy = self.entropy(y)
        best_gain = -np.inf
        best_threshold = None

        # get threshold that yields best gain
        for threshold in thresholds:
            left_partition_mask = X[attribute] <= threshold
            right_partition_mask = X[attribute] > threshold
            y_left = y[left_partition_mask]
            y_right = y[right_partition_mask]

            part_entropy = (len(y_left) / len(y)) * self.entropy(y_left)
            part_entropy += (len(y_right) / len(y)) * self.entropy(y_right)
            gain = orig_entropy - part_entropy

            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold

        return best_gain, best_threshold

    def information_gain(self, X, y, attribute):
        # categorical information gain
        orig_entropy = self.entropy(y)
        total = len(y)
        values = X[attribute].unique()
        part_entropy = 0

        for v in values:
            y_v = y[X[attribute] == v]
            if len(y_v) == 0: # shouldn't happen but just in case, skip empty splits
                continue
            part_entropy += (len(y_v) / total) * self.entropy(y_v)

        return orig_entropy - part_entropy

    def gini_numeric(self, X, y, attribute):
        unique_values = np.sort(X[attribute].unique())

        # calculate the thresholds, the midpoints between consecutive unique values
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        best_gini = np.inf
        best_threshold = None

        # get threshold that yields lowest gini
        for threshold in thresholds:
            left_partition_mask = X[attribute] <= threshold
            right_partition_mask = X[attribute] > threshold
            y_left = y[left_partition_mask]
            y_right = y[right_partition_mask]

            part_gini = 0
            # avoid empty split which would cause division by zero in the class_probabilities function
            if len(y_left) > 0: 
                class_prob_left = self.class_probabilities(y_left)
                part_gini += (len(y_left) / len(y)) * (1 - np.sum(class_prob_left**2))

            if len(y_right) > 0:
                class_prob_right = self.class_probabilities(y_right)
                part_gini += (len(y_right) / len(y)) * (1 - np.sum(class_prob_right**2))

            if part_gini < best_gini:
                best_gini = part_gini
                best_threshold = threshold

        return best_gini, best_threshold

    def gini(self, X, y, attribute):
        # categorical gini
        total = len(y)
        values = X[attribute].unique()
        gini = 0

        for v in values:
            y_v = y[X[attribute] == v]
            if len(y_v) == 0: # shouldn't happen but just in case, skip empty splits
                continue
            class_prob_v = self.class_probabilities(y_v)
            gini += (len(y_v) / total) * (1 - np.sum(class_prob_v**2))

        return gini

    def get_best_attribute(self, X, y, attributes):
        best_attr = None
        best_threshold = None
        best_is_numeric = 0

        if self.criterion == "information_gain":
            best_gain = -np.inf

            # evaluate each attribute in the random subset and find the one with the best gain
            for attr in attributes:
                # if the attribute is numerical, find the best threshold and gain for that attribute
                if self.is_numeric_attribute(X, attr):
                    gain, threshold = self.information_gain_numeric(X, y, attr)
                    is_numeric = 1

                # otherwise, just calculate gain like we did originally
                else:
                    gain = self.information_gain(X, y, attr)
                    threshold = None
                    is_numeric = 0

                # keep track of the attribute with best gain
                if gain > best_gain:
                    best_gain = gain
                    best_attr = attr
                    best_threshold = threshold
                    best_is_numeric = is_numeric
        
        elif self.criterion == "gini":
            best_gini = np.inf

            # evaluate each attribute in the random subset and find the one with the best gini
            for attr in attributes:
                # if the attribute is numerical, find the best threshold and gini for that attribute
                if self.is_numeric_attribute(X, attr):
                    gini_score, threshold = self.gini_numeric(X, y, attr)
                    is_numeric = 1

                # otherwise, just calculate gini like we did originally
                else:
                    gini_score = self.gini(X, y, attr)
                    threshold = None
                    is_numeric = 0

                # keep track of the attribute with lowest gini
                if gini_score < best_gini:
                    best_gini = gini_score
                    best_attr = attr
                    best_threshold = threshold
                    best_is_numeric = is_numeric
        else:
            raise ValueError("Invalid criterion. Use 'information_gain' or 'gini'.")

        return best_attr, best_threshold, best_is_numeric

    def get_majority_class(self, y):
        return y.mode().iloc[0]
    
    def is_majority(self, y):
        return y.value_counts().iloc[0] / len(y) >= 0.85

    def build_tree(self, X, y, attributes, depth):
        # stopping criteria - base cases for the recursion
        # 1: check if all instances in D belong to the same class
        if y.nunique() == 1:
            return TreeNode(is_leaf=True, prediction=y.iloc[0])

        # 2: check if no attributes left OR node is geq max_depth, then majority class leaf
        # 3: additional stopping criteria to prevent tree from becoming too deep
        if (len(attributes) == 0) or (self.max_depth is not None and depth >= self.max_depth) or (self.heuristic and self.is_majority(y)):
            return TreeNode(is_leaf=True, prediction=self.get_majority_class(y))

        splittable_attrs = [attr for attr in attributes if X[attr].nunique() > 1]
        # 4: if no attributes can be split on, return leaf with majority class prediction
        if len(splittable_attrs) == 0:
            return TreeNode(is_leaf=True, prediction=self.get_majority_class(y))

        best_attr, best_threshold, best_is_numeric = self.get_best_attribute(X, y, splittable_attrs)
        # 5: if no attribute provides positive gain, return leaf with majority class prediction
        if best_attr is None:
            return TreeNode(is_leaf=True, prediction=self.get_majority_class(y))

        # create the decision node
        node = TreeNode(
            attribute=best_attr,
            is_leaf=False,
            prediction=self.get_majority_class(y),
            threshold=best_threshold,
            is_numeric=best_is_numeric
        )

        # create the child nodes for numeric case
        if node.is_numeric == 1:
            left_partition_mask = X[best_attr] <= best_threshold
            right_partition_mask = X[best_attr] > best_threshold
            X_left, y_left = X[left_partition_mask], y[left_partition_mask]
            X_right, y_right = X[right_partition_mask], y[right_partition_mask]
            remaining_attrs = [attr for attr in attributes if attr != best_attr]

            node.children["leq"] = self.build_tree(X_left, y_left, remaining_attrs, depth + 1)
            node.children["gt"] = self.build_tree(X_right, y_right, remaining_attrs, depth + 1)

        # create the child nodes for categorical case
        else:
            remaining_attrs = [attr for attr in attributes if attr != best_attr]
            # create subtree for each value of the attribute
            for v in self.all_attr_val[best_attr]:
                X_v = X[X[best_attr] == v]
                y_v = y[X[best_attr] == v]

                # if empty split, make leaf with majority class of parent node
                if len(y_v) == 0:
                    node.children[v] = TreeNode(is_leaf=True, prediction=self.get_majority_class(y))
                else:
                    node.children[v] = self.build_tree(X_v, y_v, remaining_attrs, depth + 1)

        return node
    
    def predict_one(self, x, node):
        if node.is_leaf: # base case
            return node.prediction

        # get the attribute we want to test from the data point
        # and based on its attribute value, recursively call
        # predict_one until we reach a leaf node
        if node.is_numeric == 1:
            if x[node.attribute] <= node.threshold:
                next_node = node.children.get("leq")
            else:
                next_node = node.children.get("gt")
        else:
            value = x[node.attribute]
            next_node = node.children.get(value)

        # if we encounter an unseen attribute value, return majority class of current node
        if next_node is None:
            return node.prediction

        return self.predict_one(x, next_node)

    def predict(self, X):
        return np.array([self.predict_one(row, self.root) for _, row in X.iterrows()])
    