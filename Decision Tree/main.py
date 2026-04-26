import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from decision_trees import DecisionTree


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
images_dir = os.path.join(root_dir, "images")
results_dir = os.path.join(root_dir, "results")
datasets_dir = os.path.join(root_dir, "datasets")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

def load_digits_data():
    X_digits, y_digits = datasets.load_digits(return_X_y=True)
    X_raw = pd.DataFrame(X_digits)
    y_raw = pd.Series(y_digits)
    return X_raw, y_raw

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return accuracy, weighted_f1

def load_csv_data(filepath):
    full_path = os.path.join(datasets_dir, filepath)
    df = pd.read_csv(full_path, header=0)
    y_raw = df.iloc[:, -1]
    X_raw = df.iloc[:, :-1]
    return X_raw, y_raw

def evaluate_setting_cv(X, y, params, k=10, random_state=42):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    fold_accuracies = []
    fold_f1_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        tree = DecisionTree(criterion=params["criterion"],
                            heuristic=params["heuristic"],
                            max_depth=params["max_depth"]
        )
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)

        fold_accuracy, fold_f1 = calculate_metrics(y_test, y_pred)
        fold_accuracies.append(fold_accuracy)
        fold_f1_scores.append(fold_f1)

        print(
            f"    Fold {fold_idx}: Accuracy = {fold_accuracy:.4f}, "
            f"Weighted F1 = {fold_f1:.4f}"
        )

    fold_accuracies = np.array(fold_accuracies)
    fold_f1_scores = np.array(fold_f1_scores)

    return {
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "mean_f1": float(np.mean(fold_f1_scores)),
        "std_f1": float(np.std(fold_f1_scores)),
    }


def find_best_hyperparameters(X, y, hyperparameter_grid, k=10):
    rows = []

    for params in hyperparameter_grid:
        print(f"  Evaluating params: {params}")
        metrics = evaluate_setting_cv(X, y, params, k=k)
        row = {
            **params,
            **metrics,
        }
        rows.append(row)
        print(
            f"  => Accuracy = {metrics['mean_accuracy']:.4f} +/- {metrics['std_accuracy']:.4f}, "
            f"F1 = {metrics['mean_f1']:.4f} +/- {metrics['std_f1']:.4f}"
        )

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(
        by=["mean_f1", "mean_accuracy"], ascending=[False, False]
    ).reset_index(drop=True)
    return results_df

def evaluate_best_setting_trials(X, y, params, repeats=100, test_size=0.2):
    train_accuracies = []
    test_accuracies = []

    for seed in range(repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=y,
        )

        tree = DecisionTree(
            criterion=params["criterion"],
            heuristic=params["heuristic"],
            max_depth=params["max_depth"],
        )
        tree.fit(X_train, y_train)
        y_train_pred = tree.predict(X_train)
        y_test_pred = tree.predict(X_test)

        train_accuracy, _ = calculate_metrics(y_train, y_train_pred)
        test_accuracy, _ = calculate_metrics(y_test, y_test_pred)
        train_accuracies.append(float(train_accuracy))
        test_accuracies.append(float(test_accuracy))
        print(
            f"    Trial {seed + 1}: Train Accuracy = {train_accuracies[-1]:.4f}, "
            f"Test Accuracy = {test_accuracies[-1]:.4f}"
        )

    return np.array(train_accuracies), np.array(test_accuracies)


datasets_to_run = {
    "Credit": {"load_func": lambda: load_csv_data("credit_approval.csv")},
    "Parkinsons": {"load_func": lambda: load_csv_data("parkinsons.csv")},
    "Rice": {"load_func": lambda: load_csv_data("rice.csv")},
    "Digits": {"load_func": load_digits_data},
}

hyperparameter_grid = [
    {"criterion": "information_gain", "heuristic": False, "max_depth": 5},
    {"criterion": "information_gain", "heuristic": False, "max_depth": 10},
    {"criterion": "information_gain", "heuristic": True, "max_depth": 5},
    {"criterion": "information_gain", "heuristic": True, "max_depth": 10},
    {"criterion": "gini", "heuristic": False, "max_depth": 5},
    {"criterion": "gini", "heuristic": False, "max_depth": 10},
    {"criterion": "gini", "heuristic": True, "max_depth": 5},
    {"criterion": "gini", "heuristic": True, "max_depth": 10}
]

K_FOLDS = 10
REPEAT_TRIALS = 100

for dataset_name, config in datasets_to_run.items():
    print(f"\nRunning Decision Tree experiments for {dataset_name}...")

    try:
        X_raw, y_raw = config["load_func"]()
    except FileNotFoundError as exc:
        print(f"Skipping {dataset_name}: {exc}")
        continue

    print(f"\n--- Hyperparameter Search for {dataset_name} ---")
    results_df = find_best_hyperparameters(X_raw, y_raw, hyperparameter_grid, k=K_FOLDS)
    results_df["criterion"] = results_df["criterion"].astype(str)
    results_df["heuristic"] = results_df["heuristic"].astype(bool)

    results_filename = os.path.join(results_dir, f"decision_tree_{dataset_name.lower()}_cv_results.csv")
    results_df.to_csv(results_filename, index=False)
    print(f"Saved CV table to: {results_filename}")

    print(f"\nBest settings for {dataset_name}:")
    print(
        results_df[
            ["criterion", "heuristic", "max_depth", "mean_accuracy", "std_accuracy", "mean_f1", "std_f1"]
        ].to_string(index=False)
    )

    best_row = results_df.iloc[0]
    best_params = {
        "criterion": best_row["criterion"],
        "heuristic": bool(best_row["heuristic"]),
        "max_depth": int(best_row["max_depth"]),
    }

    print(f"\nBest Hyperparameters for {dataset_name}: {best_params}")

    print(f"\n--- Repeated Trials for Best Hyperparameter Setting ({dataset_name}) ---")
    train_acc, test_acc = evaluate_best_setting_trials(
        X_raw,
        y_raw,
        best_params,
        repeats=REPEAT_TRIALS,
        test_size=0.2,
    )

    train_mean = np.mean(train_acc)
    train_std = np.std(train_acc)
    train_min = np.min(train_acc)
    train_max = np.max(train_acc)
    test_mean = np.mean(test_acc)
    test_std = np.std(test_acc)
    test_min = np.min(test_acc)
    test_max = np.max(test_acc)
    
    print(f"Average training accuracy over repeated trials: {train_mean:.4f}")
    print(f"Std training accuracy over repeated trials: {train_std:.4f}")
    print(f"Minimum training accuracy: {train_min:.4f}")
    print(f"Maximum training accuracy: {train_max:.4f}")
    print(f"Average testing accuracy over repeated trials: {test_mean:.4f}")
    print(f"Std testing accuracy over repeated trials: {test_std:.4f}")
    print(f"Minimum testing accuracy: {test_min:.4f}")
    print(f"Maximum testing accuracy: {test_max:.4f}")

    plt.figure(figsize=(8, 5))
    plt.hist(train_acc, bins=20, color="lightgreen", edgecolor="black")
    plt.xlabel("Accuracy")
    plt.ylabel("Accuracy Frequency on Training Data")
    plt.title(f"Training Accuracy Histogram ({dataset_name}), criterion = {best_params['criterion']}, heuristic = {best_params['heuristic']}\n"
              f"Mean = {train_mean:.4f}, Std = {train_std:.4f}, "
              f"Min = {train_min:.4f}, Max = {train_max:.4f}")
    plt.tight_layout()
    train_accuracy_hist_filename = os.path.join(images_dir, f"decision_tree_{dataset_name.lower()}_training_accuracy_hist.png")
    plt.savefig(train_accuracy_hist_filename)
    print(f"Saved training accuracy histogram to: {train_accuracy_hist_filename}")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(test_acc, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Accuracy")
    plt.ylabel("Accuracy Frequency on Testing Data")
    plt.title(f"Testing Accuracy Histogram ({dataset_name}), criterion = {best_params['criterion']}, heuristic = {best_params['heuristic']}\n"
              f"Mean = {test_mean:.4f}, Std = {test_std:.4f}, "
              f"Min = {test_min:.4f}, Max = {test_max:.4f}")
    plt.tight_layout()
    test_accuracy_hist_filename = os.path.join(images_dir, f"decision_tree_{dataset_name.lower()}_test_accuracy_hist.png")
    plt.savefig(test_accuracy_hist_filename)
    print(f"Saved testing accuracy histogram to: {test_accuracy_hist_filename}")
    plt.close()

