import numpy as np
import decision_trees

class RandomForest:
    def __init__(self, ntree=10, max_depth=10, min_gain=0.01, min_size_for_split=5):
        self.ntree = ntree
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.min_size_for_split = min_size_for_split
        self.trees = []

    def train(self, X, y, attributes):
        self.trees = []
        n_samples = X.shape[0]

        for _ in range(self.ntree):
            # sample with replacement (bootstrapping to construct each decision tree)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # build the tree
            tree = decision_trees.build_tree(
                X_bootstrap, 
                y_bootstrap, 
                attributes, 
                depth=0, 
                max_depth=self.max_depth, 
                min_gain=self.min_gain, 
                min_size_for_split=self.min_size_for_split
            )
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            sample = X[i]
            
            # get predictions from every tree in the forest
            tree_preds = [decision_trees.traverse(tree, sample) for tree in self.trees]
            
            # filter out None predictions from unseen categories
            valid_preds = [p for p in tree_preds if p is not None]
            
            if valid_preds:
                # find the most common prediction among valid ones
                vals, counts = np.unique(valid_preds, return_counts=True)
                majority_vote = vals[np.argmax(counts)]
            else:
                # worst case if all trees returned None
                majority_vote = 0
                
            predictions.append(majority_vote)
            
        return np.array(predictions)
        
def calculate_metrics(y_true, y_pred, positive_label):
    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    tn = np.sum((y_true != positive_label) & (y_pred != positive_label))
    fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
    fn = np.sum((y_true == positive_label) & (y_pred != positive_label))

    accuracy = (tp + tn) / len(y_true) if (tp + tn) > 0 else 0 # how many of our predictions are actually correct
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0 # out of all the times we predicted positive, how many were correct
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0 # out of all the actual positives, how many did we predict correctly
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 # harmonic mean of precision and recall

    return accuracy, precision, recall, f1

def stratified_k_fold(X, y, k=5):
    # get possible labels
    classes = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in classes}
    
    # shuffle
    for c in classes:
        np.random.shuffle(class_indices[c])

    # create k folds
    folds = [[] for _ in range(k)]
    for c in classes:
        splits = np.array_split(class_indices[c], k)
        for i in range(k):
            folds[i].extend(splits[i])

    fold_splits = [] # keeps track of each training/testing set
    
    for i in range(k):
        test_idx = folds[i]
        train_idx = []
        for j in range(k):
            if i != j:
                train_idx.extend(folds[j])
        
        fold_splits.append((np.array(train_idx), np.array(test_idx)))

    return fold_splits