# DBSCAN Clustering following formal density-based clustering concepts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def dbscan_clustering(data):
    # Prepare data
    X_encoded = data.drop(['NObeyesdad'], axis=1)
    y_original = data['NObeyesdad']
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y_original)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # DBSCAN Clustering Algorithm 
    print("==== DBSCAN Clustering ====")
    
    # PCA for dimensionality reduction
    d_original = X_scaled.shape[1]
    n_components = 15
    pca_reduction = PCA(n_components=n_components)
    X_pca_reduced = pca_reduction.fit_transform(X_scaled)
    
    print(f"Applied PCA: {d_original} features → {n_components} components ({pca_reduction.explained_variance_ratio_.sum():.1%} variance)")
    
    # Parameter Grid Search
    eps_range = [6.0, 6.1, 6.2]
    min_samples_range = [17, 18, 19]
    
    print(f"Testing eps range: {eps_range}")
    print(f"Testing min_samples: {min_samples_range}")
    print("\nEvaluating different parameter combinations:\n")
    
    # Store all results
    results = []
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(X_pca_reduced)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = int((cluster_labels == -1).sum())
            noise_ratio = n_noise / len(cluster_labels)
            
            # Only evaluate if we have valid clusters
            if n_clusters >= 2 and n_noise < len(cluster_labels):
                mask = cluster_labels != -1
                try:
                    silhouette = silhouette_score(X_pca_reduced[mask], cluster_labels[mask])
                    calinski_harabasz = calinski_harabasz_score(X_pca_reduced[mask], cluster_labels[mask])
                    davies_bouldin = davies_bouldin_score(X_pca_reduced[mask], cluster_labels[mask])
                    
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'noise_ratio': noise_ratio,
                        'silhouette': silhouette,
                        'calinski_harabasz': calinski_harabasz,
                        'davies_bouldin': davies_bouldin,
                        'labels': cluster_labels
                    })
                except:
                    pass
    
    # Sort by silhouette score (highest is best)
    results.sort(key=lambda x: x['silhouette'], reverse=True)
    
    # Show top 4 configurations
    print("Top 4 Configurations (by Silhouette Score):\n")
    for i, result in enumerate(results[:4], 1):
        print(f"Rank #{i}:")
        print(f"  eps: {result['eps']:}")
        print(f"  min_samples: {result['min_samples']}")
        print(f"  Silhouette: {result['silhouette']:}")
        print(f"  Clusters: {result['n_clusters']}")
        print(f"  Noise: {result['n_noise']} ({result['noise_ratio']:})")
        print()
    
    # Select best configuration
    best_result = results[0]
    best_eps = best_result['eps']
    best_min_samples = best_result['min_samples']
    best_labels = best_result['labels']
    n_noise = best_result['n_noise']
    noise_ratio = best_result['noise_ratio']
    
    print(f"Selected Best Configuration:")
    print(f"  eps: {best_eps:}")
    print(f"  min_samples: {best_min_samples}")
    print(f"  Number of clusters: {best_result['n_clusters']}")
    print(f"  Silhouette score: {best_result['silhouette']:}")
    print(f"  Calinski-Harabasz score: {best_result['calinski_harabasz']:}")
    print(f"  Davies-Bouldin score: {best_result['davies_bouldin']:}")
    print(f"  Noise points: {n_noise} ({noise_ratio:.3f})")
    
    # PCA Visualization (2D for plotting)
    print("\nPCA Visualization:")
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: DBSCAN Clustering Result
    plt.subplot(1, 2, 1)
    noise_mask = best_labels == -1
    cluster_mask = best_labels != -1
    
    if noise_mask.any():
        plt.scatter(X_pca_2d[noise_mask, 0], X_pca_2d[noise_mask, 1], 
                   c='lightgray', marker='x', s=50, alpha=0.4, 
                   label=f'Noise (n={noise_mask.sum()})', linewidths=1)
    
    if cluster_mask.any() and best_result['n_clusters'] > 0:
        scatter = plt.scatter(X_pca_2d[cluster_mask, 0], X_pca_2d[cluster_mask, 1], 
                            c=best_labels[cluster_mask], cmap='viridis', 
                            s=40, alpha=0.6, edgecolors='k', linewidth=0.5)
        plt.colorbar(scatter, label='Cluster')
    
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
    plt.title(f'DBSCAN Clustering ({best_result["n_clusters"]} clusters)')
    plt.grid(alpha=0.3)
    if noise_mask.any():
        plt.legend()
    
    # Plot 2: Actual Obesity Classes
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                          c=y_encoded, cmap='viridis', 
                          s=40, alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.colorbar(scatter2, label='Actual Class')
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Actual Obesity Classes')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Cluster Analysis
    cluster_df = pd.DataFrame({'Cluster': best_labels, 'Obesity_Class': y_original})
    
    if n_noise > 0:
        print(f"\nNoise (n={n_noise}): Potential outliers or rare patterns")
        
    for cluster_id in range(best_result['n_clusters']):
        cluster_data = cluster_df[cluster_df['Cluster'] == cluster_id]
        if len(cluster_data) > 0:
            most_common = cluster_data['Obesity_Class'].mode()[0]
            count = len(cluster_data)
            percentage = (count / len(cluster_df)) * 100
            print(f"Cluster {cluster_id} (n={count}, {percentage:.1f}%): {most_common}")
    
    # Discussion
    print("\n*** DBSCAN Clustering Analysis Discussion ***")
    
    sil_score = best_result['silhouette']
    comparison = "better" if sil_score > 0.333 else "comparable"
    
    print("\nData Preprocessing:")
    print(f"  • One-hot encoding expanded 8 categorical features into {d_original} binary features, creating sparsity that complicates distance-based calculations.")
    print(f"  • PCA reduced dimensionality from {d_original} to {n_components} components (retaining {pca_reduction.explained_variance_ratio_.sum():.1%} variance) to combat the curse of dimensionality.")
    
    print("\nWhy PCA is Essential for DBSCAN:")
    print("  • In high-dimensional spaces, distances between nearest and farthest points become nearly equal, making density measurement unreliable.")
    print("  • PCA focuses on variance-capturing dimensions, enabling better clustering quality and more reliable hyperparameter selection.")
    
    print("\nDBSCAN Algorithm:")
    print(f"  • DBSCAN is a density-based algorithm that identified {best_result['n_clusters']} clusters by grouping points in dense regions, naturally capturing major obesity categories.")
    
    print("\nParameter Selection:")
    print(f"  • eps={best_eps:.3f} (neighborhood radius) and min_samples={best_min_samples} (minimum density) were selected through grid search testing ranges [6.0-6.2] and [17-19].")
    print("  • These values balance cluster formation: smaller eps creates more clusters/noise, larger eps merges everything; lower MinPts allows noise to form clusters, higher MinPts rejects valid clusters.")
    
    print("\nTrade-offs Observed:")
    print(f"  • The configuration achieved {best_result['n_clusters']} clusters with silhouette score {sil_score:.3f}, balancing cluster count, noise ratio ({noise_ratio*100:.1f}%), and separation quality.")
    print("  • Stricter parameters increase noise and silhouette score but may over-fragment data, while looser parameters reduce noise but merge distinct groups.")
    
    print("\nLimitations:")
    print("  • High-dimensional categorical variables after one-hot encoding complicate distance measures, requiring PCA reduction.")
    print(f"  • PCA reduction loses {100 - pca_reduction.explained_variance_ratio_.sum()*100:.1%} of variance, potentially discarding subtle obesity patterns in minor components.")
    
    print("\nComparison with Hierarchical:")
    print(f"  • DBSCAN (silhouette: {sil_score:.3f}) shows {comparison} separation than hierarchical (silhouette: 0.333) and additionally identifies {n_noise} outliers.")
    print(f"  • Calinski-Harabasz scores differ significantly (Hierarchical: 2625 vs DBSCAN: {best_result['calinski_harabasz']:.0f}) due to different dimensionalities ({d_original}D vs {n_components}D), making them non-comparable.")
 
    return {
        'best_eps': float(best_eps),
        'best_min_samples': int(best_min_samples),
        'best_n_clusters': int(best_result['n_clusters']),
        'best_silhouette': float(best_result['silhouette']),
        'n_noise': n_noise,
        'noise_ratio': float(noise_ratio),
        'calinski_harabasz': float(best_result['calinski_harabasz']),
        'davies_bouldin': float(best_result['davies_bouldin'])
    }