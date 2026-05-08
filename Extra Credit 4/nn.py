import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as sk_datasets

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
datasets_dir = os.path.join(root_dir, 'datasets')

def load_digits_data():
    digits = sk_datasets.load_digits(return_X_y=True)
    return digits[0], digits[1]


def load_csv_data(filepath):
    full_path = os.path.join(datasets_dir, filepath)
    df = pd.read_csv(full_path, header=0)
    y_raw = df.iloc[:, -1].to_numpy()
    # one hot encode categorical
    X_df = pd.get_dummies(df.iloc[:, :-1])
    X_raw = X_df.to_numpy(dtype=float)
    return X_raw, y_raw


def normalize(X_train, X_test):
    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)
    rng = maxs - mins
    # avoid divide by 0 for constant features (one hot columns in a fold)
    rng[rng == 0] = 1
    X_train_norm = 2 * ((X_train - mins) / rng) - 1  # normalize to [-1, 1]
    X_test_norm  = 2 * ((X_test  - mins) / rng) - 1
    return X_train_norm, X_test_norm


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

class NeuralNetwork:
    def __init__(self, layer_sizes, lam=0.0, learning_rate=0.1, dropout_rate=0.0):
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes)
        self.lam = lam
        self.a = learning_rate
        self.dropout_rate = dropout_rate

        self.Thetas = []
        for k in range(self.L - 1):
            # theta mat shape should be (out x in+1), first col is bias weights
            theta_mat = np.random.randn(layer_sizes[k+1], layer_sizes[k] + 1)
            self.Thetas.append(theta_mat)
    
    def add_bias_col(self, a):
        num_rows = a.shape[0]
        ones_column = np.ones((num_rows, 1))
        # 1*bias + w1*x1 + ... wn*xn
        return np.hstack([ones_column, a])

    def sigmoid(self, z): 
        return 1 / (1 + np.exp(-z))

    def forward(self, X, training=True):
        a = X
        activations = [X]       # activations after dropout
        pre_drop_acts = [X]  # activations before dropout (for backprop sigmoid deriv)
        masks = []

        for i, Theta in enumerate(self.Thetas):
            z = self.add_bias_col(a) @ Theta.T
            a_sig = self.sigmoid(z)
            pre_drop_acts.append(a_sig)

            is_hidden = i < len(self.Thetas) - 1
            if training and self.dropout_rate > 0 and is_hidden:
                # scale kept units by 1/(1-p) for inverted dropdout
                # print(a_sig.shape)
                mask = (np.random.rand(*a_sig.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                a = a_sig * mask
            else:
                mask = None
                a = a_sig

            masks.append(mask)
            activations.append(a)

        return activations, pre_drop_acts, masks

    def cost(self, X, y):
        n = X.shape[0]
        f_x = self.forward(X, training=False)[0][-1]   # f(x)=A^L
        J = -np.sum(y*np.log(f_x) + (1-y)*np.log(1-f_x)) / n

        # S = (lambda/2n) * sum of squared non bias weights
        S = 0
        for theta_mat in self.Thetas:
            S += np.sum(theta_mat[:, 1:] ** 2)
        S = (self.lam/(2*n)) * S
        return J + S

    def backprop(self, X, y):
        n = X.shape[0]

        A, A_pre, masks = self.forward(X, training=True)

        dE_dZ = [None] * self.L
        dE_dZ[-1] = A[-1] - y

        # backprop deltas: dE_dZk from dE_dZ(k+1)
        for k in range(self.L - 2, 0, -1):
            dE_dAk = dE_dZ[k + 1] @ self.Thetas[k][:, 1:]
            
            # use pre-dropout activation for correct sigmoid derivative
            dA_dZk = A_pre[k] * (1 - A_pre[k])

            dE_dZ[k] = dE_dAk * dA_dZk
            if masks[k-1] is not None:
                # 0 out grads thru dropped neurons
                dE_dZ[k] = dE_dZ[k] * masks[k-1]

        grads = []

        for k in range(self.L - 1):
            Ak_bias = self.add_bias_col(A[k])
            dE_dThetak = (1/n) * dE_dZ[k + 1].T @ Ak_bias
            
            dE_dThetak[:, 1:] += (1/n) * self.lam * self.Thetas[k][:, 1:]
            grads.append(dE_dThetak)


        for k in range(self.L - 1):
            self.Thetas[k] -= self.a * grads[k]

        return grads, dE_dZ

    def train(self, X, y, epochs=300):
        losses = []
        for e in range(epochs):
            self.backprop(X, y)
            J = self.cost(X, y)
            losses.append(J)
            # if e%100==0:
            #     print(f"epoch {e}: J={J}")
        return losses

    def predict(self, X):
        f_x = self.forward(X, training=False)[0][-1]
        return np.argmax(f_x, axis=1)

def get_kfolds(data, k=5):
    folds = [[] for _ in range(k)]
    for val in data["class"].unique():
        subset = data[data["class"] == val].reset_index(drop=True)
        subset = subset.sample(frac=1).reset_index(drop=True)
        for i, idx in enumerate(np.array_split(np.arange(len(subset)), k)):
            folds[i].append(subset.iloc[idx])
    return [pd.concat(fold).reset_index(drop=True) for fold in folds]


def labels_to_onehot(y, classes):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx = np.array([class_to_idx[c] for c in y])
    return np.eye(len(classes))[idx]


def cross_validate(X, y, layers, lam=0.01, dropout_rate=0.0, a=0.1, epochs=300):
    classes = np.unique(y)
    k = 5
    df = pd.DataFrame(X)
    df["class"] = y
    folds = get_kfolds(df, k=k)

    accs, f1s = [], []
    for fold_idx in range(k):
        test_df  = folds[fold_idx]
        train_df = pd.concat([folds[j] for j in range(k) if j != fold_idx]).reset_index(drop=True)

        X_train = train_df.drop(columns=["class"]).to_numpy(dtype=float)
        y_train = train_df["class"].to_numpy()
        X_test  = test_df.drop(columns=["class"]).to_numpy(dtype=float)
        y_test  = test_df["class"].to_numpy()

        X_train, X_test = normalize(X_train, X_test)
        y_train_onehot = labels_to_onehot(y_train, classes)

        nn = NeuralNetwork(layers, lam=lam, learning_rate=a, dropout_rate=dropout_rate)
        nn.train(X_train, y_train_onehot, epochs=epochs)

        pred_idx = nn.predict(X_test)
        pred_labels = classes[pred_idx]

        acc, f1 = calc_metrics(y_test, pred_labels)
        accs.append(acc)
        f1s.append(f1)

    return np.mean(accs), np.mean(f1s)


def learning_curve(X, y, arch, lam=0.01, dropout_rate=0.1, a=0.1, epochs=300):
    classes = np.unique(y)
    n = X.shape[0]
    idx = np.random.permutation(n)

    test_n    = int(n * 0.2)
    test_idx  = idx[:test_n]
    train_idx = idx[test_n:]

    X_test, y_test = X[test_idx], y[test_idx]
    y_test_oh = labels_to_onehot(y_test, classes)

    max_train  = len(train_idx)
    sample_sizes = np.unique(np.linspace(5, max_train, 20).astype(int))
    Js = []

    for m in sample_sizes:
        sub_idx = train_idx[:m]
        X_train, y_train = X[sub_idx], y[sub_idx]
        X_train_norm, X_test_norm = normalize(X_train, X_test)
        y_train_oh = labels_to_onehot(y_train, classes)

        nn = NeuralNetwork(arch, lam=lam, learning_rate=a, dropout_rate=dropout_rate)
        nn.train(X_train_norm, y_train_oh, epochs=epochs)
        Js.append(nn.cost(X_test_norm, y_test_oh))

    return sample_sizes, Js

datasets_to_run = {
    "Digits":     {"load_func": load_digits_data},
    "Parkinsons": {"load_func": lambda: load_csv_data("parkinsons.csv")},
    "Rice":       {"load_func": lambda: load_csv_data("rice.csv")},
    "Credit":     {"load_func": lambda: load_csv_data("credit_approval.csv")},
}

loaded = {}
for name, cfg in datasets_to_run.items():
    X, y = cfg["load_func"]()
    print(f"{name}: {X.shape[0]} instances, {X.shape[1]} features, {len(np.unique(y))} classes")
    loaded[name] = (X, y)

architectures  = None
dropout_rates = [0.1, 0.2, 0.3]
lam = 0.01

all_results = {}

for name, (X, y) in loaded.items():
    if name == "Digits":
        architectures  = [[64], [128], [256], [16, 16], [32, 16], [64, 32]]
        # continue
    else:
        architectures  = [[8], [16], [8, 8], [16, 8], [8, 8, 4], [16, 16, 4]]

    n_in, n_out  = X.shape[1], len(np.unique(y))
    print(f"\n### {name} ###")
    best_f1, best_cfg = 0, None
    results = []

    for hidden in architectures:
        for dr in dropout_rates:
            layer_sizes = [n_in] + hidden + [n_out]
            acc, f1 = cross_validate(X, y, layers=layer_sizes, lam=lam, dropout_rate=dr, a=0.1, epochs=300)
            print(f"hidden={hidden}, dropout={dr:.1f}: acc={acc:.4f}, f1={f1:.4f}")
            results.append({"hidden": hidden, "dropout_rate": dr, "acc": acc, "f1": f1})
            if f1 > best_f1:
                best_f1 = f1
                best_cfg = {"hidden": hidden, "dropout_rate": dr}

    all_results[name] = {"results": results, "best": best_cfg}
    print(f"  >> Best: hidden={best_cfg['hidden']}, dropout={best_cfg['dropout_rate']:.1f}, f1={best_f1:.4f}")




fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for ax, (name, (X, y)) in zip(axes, loaded.items()):
    best = all_results[name]["best"]
    n_in, n_out  = X.shape[1], len(np.unique(y))
    arch  = [n_in] + best["hidden"] + [n_out]
    sizes, Js = learning_curve(
        X, y, arch=arch,
        lam=lam, dropout_rate=best["dropout_rate"],
        a=0.1, epochs=400
    )
    ax.plot(sizes, Js, marker='o')
    ax.set_xlabel("Training instances")
    ax.set_ylabel("J on test set")
    ax.set_title(f"{name} — hidden={best['hidden']}, dropout={best['dropout_rate']}")
    ax.grid(True)

plt.suptitle("Learning Curves  Best architecture+dropout rate per dataset (lambda=0.01)", fontsize=14)
plt.tight_layout()
plt.show()
