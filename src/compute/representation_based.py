import numpy as np
import os
import pickle
import pandas as pd
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import (
    polynomial_kernel,
    rbf_kernel,
    laplacian_kernel,
    linear_kernel,
    sigmoid_kernel
)
import kmedoids
from sklearn.metrics import pairwise_distances
from src.compute.utils import compute_correlation

SEEDS = [42, 43, 44, 45, 46]

# web navigation domain constants
WEBSITE_NAMES = [
    'allrecipes',
    'amazon',
    'apple',
    'arxiv',
    'bbc',
    'coursera',
    'dictionary.cambridge',
    'espn',
    'github',
    'google_maps',
    'google_search',
    'huggingface',
    'wolframalpha',
]
PARTITIONS = [0, 1, 2, 3, 4]

# ===========================
# ---- PAD (Proxy-A-Distance)
# ===========================

def compute_pad(x_syn_emb, x_real_emb, classifier_name="LogisticRegression"):
    """
    Compute the Proxy-A-Distance (PAD) between two sets of embeddings.

    Args:
        x_syn_emb (np.ndarray): Embeddings of synthetic data, shape (n_samples, n_features)
        x_real_emb (np.ndarray): Embeddings of real data, shape (n_samples, n_features)
        classifier_name (str): Classifier to use ("LogisticRegression", "RandomForest", "MLP")

    Returns:
        float: PAD value
    """
    y_syn_train = np.zeros(len(x_syn_emb))
    y_real_train = np.ones(len(x_real_emb))
    x_train = np.concatenate([x_syn_emb, x_real_emb], axis=0)
    y_train = np.concatenate([y_syn_train, y_real_train], axis=0)

    # split into train/validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    # classifier
    if classifier_name == "LogisticRegression":
        classifier = LogisticRegression()
    elif classifier_name == "RandomForest":
        classifier = RandomForestClassifier()
    elif classifier_name == "MLP":
        classifier = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=200, random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    classifier.fit(x_train, y_train)
    y_pred_proba = classifier.predict_proba(x_val)[:, 1]
    average_loss = np.mean(np.abs(y_pred_proba - y_val))
    return 2 * (1 - 2 * average_loss)

# =======================================
# ---- MMD (Maximum Mean Discrepancy)
# =======================================

DEGREE = 3
GAMMA = None
COEF0 = 1

def compute_mmd(X, Y, kernel="polynomial", degree=3, gamma=None, coef0=1):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two samples: X and Y.

    Args:
        X (np.ndarray): First sample, shape (n_samples_X, n_features)
        Y (np.ndarray): Second sample, shape (n_samples_Y, n_features)
        kernel (str): Kernel name ("polynomial", "rbf", "laplacian", "linear", "sigmoid")
        degree (int): Degree for polynomial kernel (default: 3)
        gamma (float): Gamma parameter for kernels (default: None)
        coef0 (float): Coef0 for polynomial/sigmoid kernel

    Returns:
        float: MMD value
    """
    kernel = kernel.lower() if isinstance(kernel, str) else kernel
    if kernel == "polynomial":
        kfunc = polynomial_kernel
        XX = kfunc(X, X, degree=degree, gamma=gamma, coef0=coef0)
        YY = kfunc(Y, Y, degree=degree, gamma=gamma, coef0=coef0)
        XY = kfunc(X, Y, degree=degree, gamma=gamma, coef0=coef0)
    elif kernel == "rbf":
        kfunc = rbf_kernel
        XX = kfunc(X, X, gamma=gamma)
        YY = kfunc(Y, Y, gamma=gamma)
        XY = kfunc(X, Y, gamma=gamma)
    elif kernel == "laplacian":
        kfunc = laplacian_kernel
        XX = kfunc(X, X, gamma=gamma)
        YY = kfunc(Y, Y, gamma=gamma)
        XY = kfunc(X, Y, gamma=gamma)
    elif kernel == "linear":
        kfunc = linear_kernel
        XX = kfunc(X, X)
        YY = kfunc(Y, Y)
        XY = kfunc(X, Y)
    elif kernel == "sigmoid":
        kfunc = sigmoid_kernel
        XX = kfunc(X, X, gamma=gamma, coef0=coef0)
        YY = kfunc(Y, Y, gamma=gamma, coef0=coef0)
        XY = kfunc(X, Y, gamma=gamma, coef0=coef0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)

# ===============================================
# ---- Medoid-based distance (Mean Distance to Medoid)
# ===============================================

def compute_distance_matrix(embeddings, metric='euclidean'):
    """
    Compute the pairwise distance matrix for a set of embeddings.

    Args:
        embeddings (np.ndarray): Embedding matrix of shape (n_samples, n_features)
        dummy_placeholder (any): Dummy placeholder to match the signature of the function
        metric (str): Distance metric to use ('euclidean', 'cosine', etc.)

    Returns:
        np.ndarray: Distance matrix of shape (n_samples, n_samples)
    """
    return pairwise_distances(embeddings, metric=metric)

def compute_mdm(embeddings, dummy_placeholder, n_clusters=5, metric='euclidean'):
    """
    Compute the mean distance of points in each cluster to its medoid, then average across clusters,
    using the kmedoids package (fasterpam or pam).

    Args:
        embeddings (np.ndarray): Embedding matrix of shape (n_samples, n_features)
        dummy_placeholder (any): Dummy placeholder to match the signature of the function
        n_clusters (int): Number of clusters/medoids to use
        metric (str): Distance metric for KMedoids

    Returns:
        float: Mean distance to medoid (averaged over all clusters)
    """
    n_samples = len(embeddings)
    if n_samples < n_clusters:
        n_clusters = max(1, n_samples)
    diss = compute_distance_matrix(embeddings, metric=metric)
    pam_result = kmedoids.fasterpam(diss, n_clusters, random_state=42)
    labels = pam_result.labels
    medoid_indices = pam_result.medoids

    total_dist = 0.0
    for i, medoid_idx in enumerate(medoid_indices):
        cluster_points_idx = np.where(labels == i)[0]
        if len(cluster_points_idx) == 0:
            continue
        dists = diss[cluster_points_idx, medoid_idx]
        total_dist += np.mean(dists)
    return total_dist / n_clusters

def get_method_fn(method: str):
    if method == "pad":
        return compute_pad
    elif method == "mmd":
        return compute_mmd
    elif method == "mdm":
        return compute_mdm
    elif method == "mauve":
        raise ValueError(f"Method {method} not supported")

def main(args):
    results = []
    synth_embeddings = pickle.load(open(os.path.join(args.embedding_path, "synth_embeddings.pkl"), "rb")) #
    real_embeddings = pickle.load(open(os.path.join(args.embedding_path, "real_embeddings.pkl"), "rb"))
    method_fn = get_method_fn(args.method)
    if args.task == "sentiment_analysis":
        for seed in SEEDS:
            real_data_embs = real_embeddings[str(seed)]
            for dataset_name in synth_embeddings.keys():
                synth_embs = synth_embeddings[dataset_name]
                scores = method_fn(synth_embs, real_data_embs)
                results.append({
                    "seed": seed,
                    "dataset_name": dataset_name,
                    f"{args.method}_score": scores,
                })
    elif args.task == "text2sql":
        for seed in SEEDS:
            for db_id in real_embeddings[str(seed)].keys():
                real_embs = real_embeddings[str(seed)][db_id]
                for dataset_name in synth_embeddings[db_id].keys():
                    synth_embs = synth_embeddings[db_id][dataset_name]
                    scores = method_fn(synth_embs, real_embs)
                    results.append({
                        "seed": seed,
                        "db_id": db_id,
                        "dataset_name": dataset_name,
                        f"{args.method}_score": scores,
                    })
    elif args.task == "web":
        # For each website
        # TODO: find the embeddings that have seed for 
        for website in WEBSITE_NAMES:
            for seed in SEEDS:
                real_embs = real_embeddings[seed][website]
                for partition in PARTITIONS:
                    synth_embs = synth_embeddings[website][partition]
                    scores = method_fn(synth_embs, real_embs)
                    results.append({
                        "website": website,
                        "seed": seed,
                        "partition": partition,
                        f"{args.method}_score": scores,
                    })
    df_results = pd.DataFrame(results)
    df_corr_results = compute_correlation(df_results, args.method, args.task, args.task_performance_path)
    df_corr_results.to_csv(os.path.join(args.output_path, f"{args.task}_{args.method}_correlation.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to the embedding files")
    parser.add_argument("--task_performance_path", type=str, help="Path to the task performance csv file")
    parser.add_argument("--method", type=str, required=True, choices=["pad", "mmd", "mdm", "mauve"], help="Method to use")
    parser.add_argument("--task", type=str, required=True, choices=["sentiment_analysis", "text2sql", "web", "image"], help="Name of the task")
    parser.add_argument("--output_path", type=str, default="./results", help="Path to save the results")
    args = parser.parse_args()

    main(args)