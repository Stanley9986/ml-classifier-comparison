import os
import sys
import numpy as np
import pandas as pd
from neural_network import NeuralNetwork
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
datasets_dir = os.path.join(root_dir, "datasets")
results_dir = os.path.join(root_dir, "results")
random_forest_dir = os.path.join(root_dir, "Random Forest")
knn_dir = os.path.join(root_dir, "KNN")

os.makedirs(results_dir, exist_ok=True)

# import our random forest and knn implementations from their folders
sys.path.insert(0, random_forest_dir)
from random_forest import RandomForest
sys.path.insert(0, knn_dir)
from knn import knn_predict, normalize

def load_digits_data():
    X_digits, y_digits = datasets.load_digits(return_X_y=True)
    X_raw = pd.DataFrame(X_digits)
    y_raw = pd.Series(y_digits)
    return X_raw, y_raw

def load_csv_data(filepath):
    full_path = os.path.join(datasets_dir, filepath)
    df = pd.read_csv(full_path, header=0)
    y_raw = df.iloc[:, -1]
    X_raw = df.iloc[:, :-1]
    return X_raw, y_raw

def preprocess_features(X_train_raw, X_test_raw):
    # convert categorical feature columns into dummy columns after combining train and test
    combined = pd.concat([X_train_raw, X_test_raw], axis=0)
    combined = pd.get_dummies(combined)

    X_train = combined.iloc[: len(X_train_raw)].to_numpy(dtype=float)
    X_test = combined.iloc[len(X_train_raw) :].to_numpy(dtype=float)

    # scale features for neural networks and knn
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def encode_labels(y_train_raw, y_test_raw):
    # convert labels to 0, 1, 2, ... so they can be used for one hot encoding later
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)
    return y_train, y_test, encoder

def one_hot(y, n_classes):
    # convert class labels into neural network output vectors
    encoded = np.zeros((len(y), n_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded

def bootstrap_sample(X, y, seed):
    rng = np.random.RandomState(seed)
    n_samples = len(X)
    # Generate random row positions with replacement to create our bootstrap sample.
    sample_indices = rng.choice(np.arange(n_samples), size=n_samples, replace=True)
    X_sample = X[sample_indices]
    y_sample = y[sample_indices]
    return X_sample, y_sample

def train_neural_network(X_train, y_train, n_classes, hidden_layers, lambda_reg, seed):
    # each model in the ensemble is trained on its own bootstrap sample
    X_boot, y_boot = bootstrap_sample(X_train, y_train, seed)
    y_boot_one_hot = one_hot(y_boot, n_classes)

    np.random.seed(seed)
    model = NeuralNetwork(
        layer_sizes=[X_train.shape[1], *hidden_layers, n_classes],
        random_seed=seed,
    )
    model.train(
        X_boot,
        y_boot_one_hot,
        learning_rate=0.1,
        lambda_reg=lambda_reg,
        max_iterations=1000,
        batch_size=64,
        verbose=False,
    )
    return model

def train_random_forest(X_train, y_train, seed, n_trees, max_depth, min_gain, min_size_for_split):
    # set the seed so the bootstrap sample and random tree splits are reproducible
    X_boot, y_boot = bootstrap_sample(X_train, y_train, seed)
    np.random.seed(seed)
    model = RandomForest(
        ntree=n_trees,
        max_depth=max_depth,
        min_gain=min_gain,
        min_size_for_split=min_size_for_split,
    )
    attributes = list(range(X_train.shape[1]))
    model.train(X_boot, y_boot, attributes)
    return model

class KNN:
    # wrapper so we can use our KNN/knn.py functions with a fit/predict interface
    def __init__(self, k_neighbors):
        self.k_neighbors = k_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        X_train_norm, X_test_norm = normalize(self.X_train, X_test)
        return knn_predict(X_train_norm, self.y_train, X_test_norm, self.k_neighbors)

def train_knn(X_train, y_train, seed, k_neighbors):
    X_boot, y_boot = bootstrap_sample(X_train, y_train, seed)
    model = KNN(k_neighbors=k_neighbors)
    model.fit(X_boot, y_boot)
    return model

def majority_vote(member_predictions):
    # get predictions from all models (rows are models, columns are samples)
    model_predictions = np.array(member_predictions)
    final_predictions = []

    # for each sample, take the majority vote across all models
    for sample_votes in model_predictions.T:
        votes, counts = np.unique(sample_votes, return_counts=True)
        final_predictions.append(votes[np.argmax(counts)])

    return np.array(final_predictions)

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return accuracy, weighted_f1

def evaluate_nn_settings_on_dataset(dataset_name, load_func):
    X_raw, y_raw = load_func()
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    # try each neural network setting using the same 10 folds
    for params in NN_SETTINGS:
        fold_metrics = []
        print(
            f"\tTesting NN hidden_layers={params['hidden_layers']}, "
            f"lambda_reg={params['lambda_reg']}"
        )

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_raw, y_raw), start=1):
            X_train_raw = X_raw.iloc[train_idx]
            X_test_raw = X_raw.iloc[test_idx]
            y_train_raw = y_raw.iloc[train_idx]
            y_test_raw = y_raw.iloc[test_idx]

            X_train, X_test = preprocess_features(X_train_raw, X_test_raw)
            y_train, y_test, label_encoder = encode_labels(y_train_raw, y_test_raw)
            num_classes = len(label_encoder.classes_)

            model = train_neural_network(
                X_train,
                y_train,
                num_classes,
                params["hidden_layers"],
                params["lambda_reg"],
                seed=RANDOM_STATE + fold_idx,
            )
            y_pred = model.predict(X_test)
            fold_metrics.append(calculate_metrics(y_test, y_pred))

        # column 0 is accuracy, column 1 is weighted F1 score
        fold_metrics = np.array(fold_metrics)
        rows.append({
            "dataset": dataset_name,
            "hidden_layers": params["hidden_layers"],
            "lambda_reg": params["lambda_reg"],
            "accuracy": np.mean(fold_metrics[:, 0]),
            "weighted_f1": np.mean(fold_metrics[:, 1]),
        })

    return rows

def select_top_neural_network_settings(nn_sweep_rows):
    # choose top two neural network settings by F1 score, breaking ties with accuracy score
    results_df = pd.DataFrame(nn_sweep_rows)
    results_df = results_df.sort_values(
        by=["weighted_f1", "accuracy"], ascending=[False, False]
    ).reset_index(drop=True)

    selected_settings = []
    for idx, row in results_df.head(2).iterrows():
        selected_settings.append({
            "name": f"nn_arch{idx + 1}",
            "hidden_layers": row["hidden_layers"],
            "lambda_reg": row["lambda_reg"],
            "seed": 101 + idx * 101,
        })

    return selected_settings

def evaluate_ensemble_on_dataset(dataset_name, load_func, nn_settings, rf_settings, knn_setting):
    X_raw, y_raw = load_func()
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    model_metrics = {}

    # train a fresh ensemble on each training fold and evaluate it on the held-out fold
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_raw, y_raw), start=1):
        print(f"\tFold {fold_idx}/{K_FOLDS}")

        X_train_raw = X_raw.iloc[train_idx]
        X_test_raw = X_raw.iloc[test_idx]
        y_train_raw = y_raw.iloc[train_idx]
        y_test_raw = y_raw.iloc[test_idx]

        X_train, X_test = preprocess_features(X_train_raw, X_test_raw)
        y_train, y_test, label_encoder = encode_labels(y_train_raw, y_test_raw)
        num_classes = len(label_encoder.classes_)
        seed_offset = fold_idx * 1000

        # two neural networks, two random forests, and one knn model
        members = []
        for nn_setting in nn_settings:
            members.append((
                nn_setting["name"],
                train_neural_network(
                    X_train,
                    y_train,
                    num_classes,
                    nn_setting["hidden_layers"],
                    nn_setting["lambda_reg"],
                    seed=nn_setting["seed"] + seed_offset,
                ),
            ))

        for rf_setting in rf_settings:
            members.append((
                rf_setting["name"],
                train_random_forest(
                    X_train,
                    y_train,
                    seed=rf_setting["seed"] + seed_offset,
                    n_trees=rf_setting["n_trees"],
                    max_depth=rf_setting["max_depth"],
                    min_gain=rf_setting["min_gain"],
                    min_size_for_split=rf_setting["min_size_for_split"],
                ),
        ))

        members.append((
            "knn",
            train_knn(
                X_train,
                y_train,
                seed=knn_setting["seed"] + seed_offset,
                k_neighbors=knn_setting["k_neighbors"],
            ),
        ))

        member_predictions = []

        for member_name, model in members:
            y_pred = model.predict(X_test)
            member_predictions.append(y_pred)
            accuracy, weighted_f1 = calculate_metrics(y_test, y_pred)
            model_metrics.setdefault(member_name, []).append((accuracy, weighted_f1))

        # combine the base model predictions using majority voting
        ensemble_pred = majority_vote(member_predictions)
        accuracy, weighted_f1 = calculate_metrics(y_test, ensemble_pred)
        model_metrics.setdefault("majority_vote_ensemble", []).append((accuracy, weighted_f1))

    # average the fold metrics for each model
    rows = []
    for model_name, metrics in model_metrics.items():
        metrics = np.array(metrics)
        rows.append({
            "dataset": dataset_name,
            "model": model_name,
            "accuracy": np.mean(metrics[:, 0]),
            "weighted_f1": np.mean(metrics[:, 1]),
        })

    return rows

# global constants 
RANDOM_STATE = 42
K_FOLDS = 10
NN_ARCHITECTURES = [
    [8],
    [16],
    [16, 8],
    [24, 12],
    [16, 8, 4],
    [32, 16, 8],
]
NN_LAMBDA_REG_VALUES = [0.01, 0.1]
NN_SETTINGS = [
    {"hidden_layers": hidden_layers, "lambda_reg": lambda_reg}
    for hidden_layers in NN_ARCHITECTURES for lambda_reg in NN_LAMBDA_REG_VALUES
]

def main():
    datasets_to_run = {
        "Credit": lambda: load_csv_data("credit_approval.csv"),
        "Parkinsons": lambda: load_csv_data("parkinsons.csv"),
        "Rice": lambda: load_csv_data("rice.csv"),
        "Digits": load_digits_data,
    }
    rf_settings_for_dataset = {
        "Credit": [
            {"name": "rf_credit_1", "n_trees": 50, "max_depth": 15, "min_gain": 0.01, "min_size_for_split": 5, "seed": 101},
            {"name": "rf_credit_2", "n_trees": 20, "max_depth": 10, "min_gain": 0.01, "min_size_for_split": 5, "seed": 102},
        ],
        "Parkinsons": [
            {"name": "rf_parkinsons_1", "n_trees": 20, "max_depth": 25, "min_gain": 0.01, "min_size_for_split": 5, "seed": 101},
            {"name": "rf_parkinsons_2", "n_trees": 20, "max_depth": 10, "min_gain": 0.01, "min_size_for_split": 5, "seed": 102},
        ],
        "Rice": [
            {"name": "rf_rice_1", "n_trees": 5, "max_depth": 25, "min_gain": 0.05, "min_size_for_split": 10, "seed": 101},
            {"name": "rf_rice_2", "n_trees": 20, "max_depth": 15, "min_gain": 0.05, "min_size_for_split": 10, "seed": 102},
        ],
        "Digits": [
            {"name": "rf_digits_1", "n_trees": 50, "max_depth": 10, "min_gain": 0.01, "min_size_for_split": 5, "seed": 101},
            {"name": "rf_digits_2", "n_trees": 20, "max_depth": 15, "min_gain": 0.01, "min_size_for_split": 5, "seed": 102},
        ],
    }
    knn_settings_for_dataset = {
        "Credit": {"k_neighbors": 13, "seed": 103},
        "Parkinsons": {"k_neighbors": 1, "seed": 103},
        "Rice": {"k_neighbors": 17, "seed": 103},
        "Digits": {"k_neighbors": 5, "seed": 103},
    }

    all_rows = []
    nn_sweep_rows = []

    for dataset_name, load_func in datasets_to_run.items():
        # first get the top 2 neural network settings for this dataset
        print(f"\nRunning neural network hyperparameter sweep on {dataset_name}...")
        dataset_nn_sweep_rows = evaluate_nn_settings_on_dataset(dataset_name, load_func)
        nn_sweep_rows.extend(dataset_nn_sweep_rows)
        nn_settings = select_top_neural_network_settings(dataset_nn_sweep_rows)

        print("\tSelected neural network settings:")
        for setting in nn_settings:
            print(
                f"\t\t{setting['name']}: hidden_layers={setting['hidden_layers']}, "
                f"lambda_reg={setting['lambda_reg']}"
            )

        # then evaluate the ensemble using the selected neural network settings
        print(f"\nRunning ensemble on {dataset_name}...")
        rows = evaluate_ensemble_on_dataset(
            dataset_name,
            load_func,
            nn_settings,
            rf_settings_for_dataset[dataset_name],
            knn_settings_for_dataset[dataset_name],
        )
        all_rows.extend(rows)

        ensemble_metrics = rows[-1]
        print(
            f"\tEnsemble accuracy={ensemble_metrics['accuracy']:.4f}, "
            f"weighted F1={ensemble_metrics['weighted_f1']:.4f}"
        )

    results_df = pd.DataFrame(all_rows)
    output_path = os.path.join(results_dir, "ensemble_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved ensemble results to: {output_path}")

    nn_sweep_df = pd.DataFrame(nn_sweep_rows)
    nn_sweep_output_path = os.path.join(results_dir, "ensemble_nn_sweep_results.csv")
    nn_sweep_df.to_csv(nn_sweep_output_path, index=False)
    print(f"Saved neural network sweep results to: {nn_sweep_output_path}")


if __name__ == "__main__":
    main()
