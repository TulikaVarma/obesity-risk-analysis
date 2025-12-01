import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class CLARANS:
    def __init__(self, n_clusters: int, num_local: int = 5, max_neighbor: int = 20, random_state: int = 42):
        self.n_clusters = n_clusters
        self.num_local = num_local
        self.max_neighbor = max_neighbor
        self.random_state = random_state
        self.medoids_ = None
        self.labels_ = None

    def _distance_matrix(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - X[None, :, :]
        return np.linalg.norm(diff, axis=2)

    def _cost(self, dist_matrix: np.ndarray, medoid_idx: np.ndarray) -> float:
        return dist_matrix[:, medoid_idx].min(axis=1).sum()

    def fit(self, X: np.ndarray):
        X = np.asarray(X)
        rng = np.random.default_rng(self.random_state)
        dist_matrix = self._distance_matrix(X)

        best_cost = np.inf
        best_medoids = None

        for _ in range(self.num_local):
            medoids = rng.choice(X.shape[0], self.n_clusters, replace=False)
            current_cost = self._cost(dist_matrix, medoids)

            for _ in range(self.max_neighbor):
                improved = False
                for i in range(len(medoids)):
                    non_medoids = np.setdiff1d(np.arange(X.shape[0]), medoids)
                    if len(non_medoids) == 0:
                        continue
                    candidate = rng.choice(non_medoids)
                    new_medoids = medoids.copy()
                    new_medoids[i] = candidate
                    new_cost = self._cost(dist_matrix, new_medoids)
                    if new_cost < current_cost:
                        medoids = new_medoids
                        current_cost = new_cost
                        improved = True
                        break
                if not improved:
                    break

            if current_cost < best_cost:
                best_cost = current_cost
                best_medoids = medoids

        self.medoids_ = best_medoids
        distances_to_medoids = dist_matrix[:, best_medoids]
        self.labels_ = np.argmin(distances_to_medoids, axis=1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.medoids_ is None:
            raise ValueError("Model not fit yet.")
        dist_matrix = self._distance_matrix(X)
        return np.argmin(dist_matrix[:, self.medoids_], axis=1)

def clarans_clustering(X_pca, y):
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    cluster_options = [3, 5, 7, 9]
    results = []

    print("==== CLARANS Clustering ====")
    print(f"Input: {X_pca.shape[1]} PCA components (pre-processed)")
    print(f"Evaluating clusters: {cluster_options}\n")

    for k in cluster_options:
        if k <= 1 or k > len(X_pca):
            continue
        clarans = CLARANS(n_clusters=k, num_local=5, max_neighbor=15, random_state=42).fit(X_pca)
        labels = clarans.labels_

        if len(np.unique(labels)) < 2:
            continue

        sil = silhouette_score(X_pca, labels)
        calinski = calinski_harabasz_score(X_pca, labels)
        davies = davies_bouldin_score(X_pca, labels)

        results.append({
            'n_clusters': k,
            'labels': labels,
            'silhouette': sil,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies
        })

        print(f"k={k}: silhouette={sil:.3f}, calinski={calinski:.1f}, davies_bouldin={davies:.3f}")

    if not results:
        print("CLARANS could not produce valid clusters.")
        return {}

    best = max(results, key=lambda r: r['silhouette'])
    print(f"\nBest configuration: k={best['n_clusters']} "
          f"(silhouette={best['silhouette']:.3f}, calinski={best['calinski_harabasz']:.1f}, "
          f"davies_bouldin={best['davies_bouldin']:.3f})")

    # PCA to 2D for plotting
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_pca)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    scatter = plt.scatter(
        X_pca_2d[:, 0], X_pca_2d[:, 1],
        c=best['labels'], cmap='tab20',
        alpha=0.6, edgecolors='k', linewidth=0.3
    )
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
    plt.title(f'CLARANS Clustering (k={best["n_clusters"]})')
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(
        X_pca_2d[:, 0], X_pca_2d[:, 1],
        c=y_encoded, cmap='tab20',
        alpha=0.6, edgecolors='k', linewidth=0.3
    )
    plt.colorbar(scatter2, label='Actual Class')
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Actual Obesity Classes')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'best_n_clusters': int(best['n_clusters']),
        'best_silhouette': float(best['silhouette']),
        'calinski_harabasz': float(best['calinski_harabasz']),
        'davies_bouldin': float(best['davies_bouldin']),
    }
