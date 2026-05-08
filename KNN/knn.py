import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from collections import Counter


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
images_dir = os.path.join(root_dir, "images")
datasets_dir = os.path.join(root_dir, "datasets")

os.makedirs(images_dir, exist_ok=True)



def load_digits_data():
    digits = datasets.load_digits(return_X_y=True)
    return digits[0], digits[1]

def load_csv_data(filepath):
    full_path = os.path.join(datasets_dir, filepath)
    df = pd.read_csv(full_path, header=0)
    y_raw = df.iloc[:, -1].to_numpy()
    # one hot encode categorical
    X_df = pd.get_dummies(df.iloc[:, :-1])
    X_raw = X_df.to_numpy(dtype=float)
    return X_raw, y_raw

datasets_to_run = {
    "Digits": {"load_func": load_digits_data},
    "Parkinsons": {"load_func": lambda: load_csv_data("parkinsons.csv")},
    "Rice": {"load_func": lambda: load_csv_data("rice.csv")},
    "Credit": {"load_func": lambda: load_csv_data("credit_approval.csv")},
}

# X, y = datasets_to_run["Credit"]["load_func"]()




def normalize(X_train, X_test):
    mins = X_train.min(axis=0)  # collapse over axis 0
    maxs = X_train.max(axis=0)
    rng = maxs - mins
    # avoid divide by 0 for features in a fold where all values are same, feature would be meaningless
    # (when using one hot encodings)
    rng[rng == 0] = 1

    X_train_norm = 2 * ((X_train - mins) / rng) - 1 # normalize to [-1, 1]
    X_test_norm = 2 * ((X_test - mins) / rng) - 1
    return X_train_norm, X_test_norm


def knn_predict(X_train, y_train, X_test, k):
    preds = []
    for x in X_test:
        # X_train:    (N, D)
        # x: (D,)     (1, D)
        # dists:  (N,)
        diffs = X_train - x
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))

        idxs = np.argsort(dists)[:k]
        neighbor_labels = y_train[idxs]

        cts = Counter(neighbor_labels).items()
        sorted_cts = sorted(cts, key=lambda x: x[1], reverse=True)
        preds.append(sorted_cts[0][0])
    return np.array(preds)


def calc_metrics(y_true, y_pred):
    classes = np.unique(y_true)
    f1_scores, total_tp = [], 0
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
        f1_scores.append(f1)
        total_tp += tp
    return total_tp / len(y_true), sum(f1_scores) / len(f1_scores)


def get_kfolds(data, k=5):
    folds = [[] for _ in range(k)]
    for val in data["class"].unique():
        subset = data[data["class"] == val].reset_index(drop=True)
        subset = subset.sample(frac=1).reset_index(drop=True)
        for i, idx in enumerate(np.array_split(np.arange(len(subset)), k)):
            folds[i].append(subset.iloc[idx])
    return [pd.concat(fold).reset_index(drop=True) for fold in folds]







def main():
    K_vals = list(range(1, 52, 2))

    best_k_per_dataset = {}
    all_results = {}

    for dataset_name, config in datasets_to_run.items():
        print(f"\n################################")
        print(f"### K search: {dataset_name} ###")
        X_raw, y_raw = config["load_func"]()

        df = pd.DataFrame(X_raw)
        df["class"] = y_raw
        folds = get_kfolds(df, k=10)

        k_acc, k_acc_std, k_f1 = [], [], []

        for k in K_vals:
            fold_acc, fold_f1 = [], []
            for i in range(10):
                test_df  = folds[i]
                train_df = pd.concat([folds[j] for j in range(10) if j != i]).reset_index(drop=True)

                X_train = train_df.drop(columns=["class"]).to_numpy(dtype=float)
                y_train = train_df["class"].to_numpy()
                X_test  = test_df.drop(columns=["class"]).to_numpy(dtype=float)
                y_test  = test_df["class"].to_numpy()

                X_train_norm, X_test_norm = normalize(X_train, X_test)
                y_pred = knn_predict(X_train_norm, y_train, X_test_norm, k)

                acc, f1 = calc_metrics(y_test, y_pred)
                fold_acc.append(acc)
                fold_f1.append(f1)

            k_acc.append(np.mean(fold_acc))
            k_acc_std.append(np.std(fold_acc))
            k_f1.append(np.mean(fold_f1))
            print(f"k={k}, acc={k_acc[-1]:.4f}, F1={k_f1[-1]:.4f}")

        best_idx = int(np.argmax(k_f1))
        best_k = K_vals[best_idx]
        best_k_per_dataset[dataset_name] = best_k
        all_results[dataset_name] = {"k_acc": k_acc, "k_acc_std": k_acc_std, "k_f1": k_f1}

        print(f"\n best k for {dataset_name}: {best_k}")

    for dataset_name, res in all_results.items():
        plt.figure(figsize=(10, 5))
        plt.errorbar(K_vals, res["k_acc"], yerr=res["k_acc_std"], marker="o", linestyle="-", capsize=4, label="Accuracy")
        plt.plot(K_vals, res["k_f1"], marker="s", linestyle="--", label="F1")
        plt.title(f"Performance vs k: {dataset_name}")
        plt.xlabel("k")
        plt.ylabel("Performance")
        plt.xticks(K_vals[::2])
        plt.legend()
        plt.grid(True)

        path = os.path.join(images_dir, f"knn_{dataset_name.lower()}_k.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
