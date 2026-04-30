import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import random_forest

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
images_dir = os.path.join(root_dir, "images")
datasets_dir = os.path.join(root_dir, "datasets")

os.makedirs(images_dir, exist_ok=True)

def calculate_macro_metrics(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    
    classes = np.unique(y_true)
    f1_scores = []
    
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        
    macro_f1 = np.mean(f1_scores)
    return accuracy, macro_f1

def load_digits_data():
    digits = datasets.load_digits(return_X_y=True)
    return digits[0], digits[1]

def load_csv_data(filepath, label_first=True):
    full_path = os.path.join(datasets_dir, filepath)
    df = pd.read_csv(full_path, header=0)
    if label_first:
        y_raw = df.iloc[:, 0].to_numpy()
        X_raw = df.iloc[:, 1:].to_numpy()
    else:
        y_raw = df.iloc[:, -1].to_numpy()
        X_raw = df.iloc[:, :-1].to_numpy()
    return X_raw, y_raw

datasets_to_run = {
    # "Digits": {"load_func": load_digits_data},
    # "Parkinsons": {"load_func": lambda: load_csv_data("parkinsons.csv", label_first=False)},
    # "Rice": {"load_func": lambda: load_csv_data("rice.csv", label_first=False)},
    # "Credit": {"load_func": lambda: load_csv_data("credit_approval.csv", label_first=False)},
    "Valorant": {"load_func": lambda: load_csv_data("valorant_dataset_500_color.csv", label_first=True)},
}

hyperparameter_grid = [
    {"max_depth": 10, "min_gain": 0.01, "min_size_for_split": 5},
    # {"max_depth": 10, "min_gain": 0.05, "min_size_for_split": 10},
    # {"max_depth": 15, "min_gain": 0.01, "min_size_for_split": 5},
    # {"max_depth": 15, "min_gain": 0.05, "min_size_for_split": 10},
    # {"max_depth": 25, "min_gain": 0.01, "min_size_for_split": 5},
    # {"max_depth": 25, "min_gain": 0.05, "min_size_for_split": 10}
]

ntree_values = [1, 5, 10, 20, 30, 50]
K_FOLDS = 10

for dataset_name, config in datasets_to_run.items():
    print(f"Running experiments for {dataset_name}...")
    try:
        X_raw, y_raw = config["load_func"]()
    except FileNotFoundError as e:
        print(f"Skipping {dataset_name}: {e}")
        continue

    attributes = list(range(X_raw.shape[1]))
    
    print(f"\n--- Hyperparameter Search for {dataset_name} ---")
    grid_results = []
    
    for params in hyperparameter_grid:
        fold_acc, fold_f1 = [], []
        for train_idx, test_idx in random_forest.stratified_k_fold(X_raw, y_raw, k=K_FOLDS):
            X_train, y_train = X_raw[train_idx], y_raw[train_idx]
            X_test, y_test = X_raw[test_idx], y_raw[test_idx]
            
            rf = random_forest.RandomForest(ntree=20, max_depth=params["max_depth"], min_gain=params["min_gain"], min_size_for_split=params["min_size_for_split"])
            rf.train(X_train, y_train, attributes)
            
            y_pred = rf.predict(X_test)
            
            acc, f1 = calculate_macro_metrics(y_test, y_pred)
            fold_acc.append(acc)
            fold_f1.append(f1)
            
        avg_acc = np.mean(fold_acc)
        avg_f1 = np.mean(fold_f1)
        grid_results.append((params, avg_acc, avg_f1))
        print(f"Params: {params} | Acc: {avg_acc:.4f} | F1: {avg_f1:.4f}")
        
    best_params_tuple = max(grid_results, key=lambda x: x[2])
    best_params = best_params_tuple[0]
    
    print(f"\nBest Params for {dataset_name}: {best_params}")
    print(f"--- Generating Learning Curve for {dataset_name} ---")
    
    curve_acc, curve_f1 = [], []
    
    for ntree in ntree_values:
        fold_acc, fold_f1 = [], []
        for train_idx, test_idx in random_forest.stratified_k_fold(X_raw, y_raw, k=K_FOLDS):
            X_train, y_train = X_raw[train_idx], y_raw[train_idx]
            X_test, y_test = X_raw[test_idx], y_raw[test_idx]
            
            rf = random_forest.RandomForest(ntree=ntree, max_depth=best_params["max_depth"], min_gain=best_params["min_gain"], min_size_for_split=best_params["min_size_for_split"])
            rf.train(X_train, y_train, attributes)
            
            y_pred = rf.predict(X_test)
            
            acc, f1 = calculate_macro_metrics(y_test, y_pred)
            fold_acc.append(acc)
            fold_f1.append(f1)
            
        curve_acc.append(np.mean(fold_acc))
        curve_f1.append(np.mean(fold_f1))
        print(f"ntree: {ntree:2d} | Acc: {curve_acc[-1]:.4f} | F1: {curve_f1[-1]:.4f}")
        
    plt.figure(figsize=(8, 5))
    plt.plot(ntree_values, curve_acc, marker="o", linestyle="-", color="teal", label="Accuracy")
    plt.plot(ntree_values, curve_f1, marker="s", linestyle="--", color="indigo", label="F1 Score")
    plt.title(f"Performance vs Number of Trees ({dataset_name})")
    plt.xlabel("Number of Trees (ntree)")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.xticks(ntree_values)
    
    plot_filename = os.path.join(images_dir, f"rf_{dataset_name.lower()}_learning_curve.png")
    plt.savefig(plot_filename)
    print(f"SAVED LEARNING CURVE TO: {plot_filename}\n")
    plt.close()