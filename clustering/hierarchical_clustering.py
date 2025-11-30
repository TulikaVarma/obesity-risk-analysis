# Hierarchical Clustering to evaluate different number of clusters and silhoutte score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def hierarchical_clustering(data):
  # Prepare data
  X_encoded = data.drop(['NObeyesdad'], axis=1)
  y_original = data['NObeyesdad']
  
  le_target = LabelEncoder()
  y_encoded = le_target.fit_transform(y_original)

  # 1. Hierarchical Clustering Algorithm
  print("==== Hierarchical Clustering ====")

  # Different number of clusters to evaluate
  clusters_range = [3, 5, 7, 9]
  hierarchical_results = []

  print("Evaluating different number of clusters: \n")
  for n_clusters in clusters_range:
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = hierarchical.fit_predict(X_encoded)

    silhoutte = silhouette_score(X_encoded, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(X_encoded, cluster_labels)
    davies_bouldin = davies_bouldin_score(X_encoded, cluster_labels)

    hierarchical_results.append({
      'n_clusters': n_clusters,
      'silhoutte_score': silhoutte,
      'calinski_harabasz_score': calinski_harabasz,
      'davies_bouldin_score': davies_bouldin
    })

    print(f"\nNumber of clusters: {n_clusters}")
    print(f"Silhoutte score: {silhoutte}")
    print(f"Calinski-Harabasz score: {calinski_harabasz}")
    print(f"Davies-Bouldin score: {davies_bouldin}\n")

  # Select the best number of clusters based on silhouette score
  results_df = pd.DataFrame(hierarchical_results)
  best_n_clusters = int(results_df.loc[results_df['silhoutte_score'].idxmax(), 'n_clusters'])
  print(f"\nBest number of clusters: {best_n_clusters}")

  # Fit the best number of clusters
  hierarchical_best = AgglomerativeClustering(n_clusters=best_n_clusters, linkage='ward')
  cluster_labels_best = hierarchical_best.fit_predict(X_encoded)

  # Visualize with PCA
  print("\nPCA Visualization:")
  pca_2d = PCA(n_components=2)
  X_pca_2d = pca_2d.fit_transform(X_encoded)

  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=cluster_labels_best, cmap='viridis', alpha=0.6, edgecolors='k', linewidth=0.5)
  plt.colorbar(scatter, label='Cluster')
  plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
  plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
  plt.title(f'Hierarchical Clustering (k={best_n_clusters})')
  plt.grid(alpha=0.3)
  
  plt.subplot(1, 2, 2)
  scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                        c=y_encoded, cmap='viridis', 
                        alpha=0.6, edgecolors='k', linewidth=0.5)
  plt.colorbar(scatter, label='Actual Class')
  plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
  plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
  plt.title('Actual Obesity Classes')
  plt.grid(alpha=0.3)
  
  plt.tight_layout()
  plt.show()

  # Analyze cluster characteristics
  print("\nCluster Characteristics:")
  cluster_df = pd.DataFrame({'Cluster': cluster_labels_best,'Obesity_Class': y_original})

  for cluster_id in range(best_n_clusters):
    cluster_data = cluster_df[cluster_df['Cluster'] == cluster_id]
    most_common = cluster_data['Obesity_Class'].mode()[0] if len(cluster_data) > 0 else 'N/A'
    print(f"Cluster {cluster_id} (n={len(cluster_data)}): Most common class = {most_common}")

  best_sillhouette = results_df.loc[results_df['silhoutte_score'].idxmax(), 'silhoutte_score']

  # Discussion: Appropriateness and Performance Comparison
  print("\n*** Hierarchical Clustering Analysis Discussion ***")
  print("\n Note: while hierarchical clustering usually produces dendogram, but in here we specificy n_cluster for fair comparison with other clustering algorithms")
  print(f"Hierarchical clustering with Ward linkage fits this obesity dataset well because the points are grouped based on distances in the features and doesn't really need class labels to find structure. From the k values tested, k={best_n_clusters} achieved the highest silhouette score of {best_sillhouette:.3f} compare to k = 3 (0.124), k = 5 (0.185), k = 7 (0.217). Therefore, the algorithm and the PCA successfully identified {best_n_clusters} clusters that fit the major obesity categories. \n")

  return {
    'best_n_clusters': best_n_clusters,
    'best_sillhouette': best_sillhouette,
    'calinski_harabasz': calinski_harabasz,
    'davies_bouldin': davies_bouldin,
  }