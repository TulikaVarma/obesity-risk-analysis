# DBSCAN Clustering with Heuristic Parameter Selection 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

def find_optimal_eps(X, min_samples, plot=True):
    # Compute k-nearest neighbors distances
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)
    
    # Get k-distance in descending order 
    k_distances = distances[:, -1]
    k_distances = np.sort(k_distances)[::-1]
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(k_distances)), k_distances, 'b-', linewidth=2)
        plt.xlabel('Points sorted by distance', fontsize=12)
        plt.ylabel(f'{min_samples}-distance', fontsize=12)
        plt.title(f'K-Distance Graph (k={min_samples})', fontsize=14)
        plt.grid(True, alpha=0.3)
        # Find elbow point using gradient method
        if len(k_distances) > 10:
            # Smooth the curve slightly
            window = min(50, len(k_distances) // 20)
            smoothed = np.convolve(k_distances, np.ones(window)/window, mode='valid')
            
            # Calculate gradient
            gradient = np.gradient(smoothed)
            gradient_change = np.gradient(gradient)
            
            # Find elbow (point of maximum curvature change)
            elbow_idx = np.argmax(np.abs(gradient_change)) + window // 2
            optimal_eps = k_distances[elbow_idx]
            
            plt.axhline(y=optimal_eps, color='r', linestyle='--', linewidth=2, 
                       label=f'Suggested eps = {optimal_eps:.5f}')
            plt.axvline(x=elbow_idx, color='r', linestyle='--', linewidth=2, alpha=0.5)
            plt.legend(fontsize=11)
        
        plt.tight_layout()
        plt.show()

    # Epsilon value 
    return optimal_eps

def dbscan_clustering(X_pca, y):
    # Get dimensionality of original feature space after hot encoding 
    d_original = X_pca.shape[1]
    
    # Encode target labels
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # DBSCAN Clustering Algorithm 
    print("\n==== DBSCAN Clustering ====")
    print(f"Input: {X_pca.shape[1]} PCA components (pre-processed)")

    # Parameter values 
    k = 2 * d_original - 1
    min_samples = k + 1
    optimal_eps = find_optimal_eps(X_pca, k, plot=True)
    heuristic_eps = optimal_eps
    print(f"Parameter Selection:")
    print(f" - d = {d_original}")
    print(f" - k = {k}")
    print(f" - MinPts = {min_samples}")
    print(f" - epsilon = {heuristic_eps:.5f}")
    
    # Run DBSCAN with heuristic parameters
    dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_pca)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    heuristic_clusters = n_clusters
    n_noise = int((cluster_labels == -1).sum())
    noise_ratio = n_noise / len(cluster_labels)
    
    # Calculate silhouette for heuristic parameters 
    heuristic_silhouette = "unavailable"
    if n_clusters >= 2 and n_noise < len(cluster_labels):
        mask = cluster_labels != -1
        try:
            heuristic_silhouette = f"{silhouette_score(X_pca[mask], cluster_labels[mask]):.3f}"
        except:
            heuristic_silhouette = "unavailable"
    
    # Print values from heurestic parameters
    print(f"Results with parameter selection eps: {optimal_eps:.5f}, clusters: {n_clusters}, noise: {n_noise}, silhouette: {heuristic_silhouette}")

    # If heuristic didn't produce valid clusters use eps with multiplier 
    if n_clusters < 2:
        print(f"\nTrying different eps values using a multiplier for more meaningful results\n")
        
        # Store results for all multipliers
        test_results = []
        # Try different epsilon values
        for multiplier in [0.3, 0.5, 0.8]:
            test_eps = optimal_eps * multiplier
            dbscan_test = DBSCAN(eps=test_eps, min_samples=min_samples)
            test_labels = dbscan_test.fit_predict(X_pca)
            test_n_clusters = len(set(test_labels)) - (1 if -1 in test_labels else 0)
            test_n_noise = int((test_labels == -1).sum())
            
            # Calculate silhouette score for valid clusters
            test_silhouette = "unavailable"
            test_silhouette_float = -1.0
            if test_n_clusters >= 2 and test_n_noise < len(cluster_labels):
                mask = test_labels != -1
                try:
                    test_silhouette_float = silhouette_score(X_pca[mask], test_labels[mask])
                    test_silhouette = f"{test_silhouette_float:.5f}"
                except:
                    test_silhouette = "unavailable"
                    test_silhouette_float = -1.0
            
            print(f"eps: {test_eps:.5f}, clusters: {test_n_clusters}, noise: {test_n_noise}, silhouette: {test_silhouette}")
            
            # Store result
            test_results.append({
                'eps': test_eps,
                'labels': test_labels,
                'n_clusters': test_n_clusters,
                'n_noise': test_n_noise,
                'silhouette': test_silhouette,
                'silhouette_float': test_silhouette_float,
                'valid': test_n_clusters >= 2 and test_n_noise < len(cluster_labels) * 0.9
            })
        
        # Select the best valid result based on silhouette score
        valid_results = [r for r in test_results if r['valid']]
        if valid_results:
            # Pick the result with the best silhouette score
            best_result = max(valid_results, key=lambda r: r['silhouette_float'])
            optimal_eps = best_result['eps']
            cluster_labels = best_result['labels']
            n_clusters = best_result['n_clusters']
            n_noise = best_result['n_noise']
            noise_ratio = n_noise / len(cluster_labels)
            best_silhouette_float = best_result['silhouette_float']
            print(f"\nSelected eps={optimal_eps:.5f} based on best silhouette score")
    
    # Calculate remaining metrics if we have valid clusters
    if n_clusters >= 2 and n_noise < len(cluster_labels):
        mask = cluster_labels != -1
        try:
            calinski_harabasz = calinski_harabasz_score(X_pca[mask], cluster_labels[mask])
            davies_bouldin = davies_bouldin_score(X_pca[mask], cluster_labels[mask])
        except:
            calinski_harabasz = 0.0
            davies_bouldin = 999.0
    else:
        print("\nNo valid clusters found")
        calinski_harabasz = 0.0
        davies_bouldin = 999.0
    
    print(f"\nDBSCAN Results:")
    print(f" - eps={optimal_eps:.5f}, min_samples={min_samples},clusters: {n_clusters}, noise: {n_noise} ({noise_ratio*100:.1f}%)")
    print(f" - Silhouette Score: {best_silhouette_float:.5f}")
    print(f" - Calinski-Harabasz Index: {calinski_harabasz:.5f}")
    print(f" - Davies-Bouldin Index: {davies_bouldin:.5f}")
    
    # Create 2D PCA for visualization 
    print("\nPCA visualization: ")
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_pca)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    # Plot 1: DBSCAN Clustering Result
    plt.subplot(1, 2, 1)
    noise_mask = cluster_labels == -1
    cluster_mask = cluster_labels != -1
    
    if noise_mask.any():
        plt.scatter(X_pca_2d[noise_mask, 0], X_pca_2d[noise_mask, 1], 
                   c='lightgray', marker='x', s=50, alpha=0.4, 
                   label=f'Noise (n={noise_mask.sum()})', linewidths=1)
    
    if cluster_mask.any() and n_clusters > 0:
        scatter = plt.scatter(X_pca_2d[cluster_mask, 0], X_pca_2d[cluster_mask, 1], 
                            c=cluster_labels[cluster_mask], cmap='viridis', 
                            s=40, alpha=0.6, edgecolors='k', linewidth=0.5)
        plt.colorbar(scatter, label='Cluster')
    
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
    plt.title(f'DBSCAN Clustering ({n_clusters} clusters)\n(PCA projection for visualization only)')
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
    plt.title('Actual Obesity Classes\n(PCA projection for visualization only)')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Cluster Analysis
    print("\nCluster Composition:")
    cluster_df = pd.DataFrame({'Cluster': cluster_labels, 'Obesity_Class': y})
    
    if n_noise > 0:
        print(f" - Noise (n={n_noise}, {noise_ratio*100:.1f}")
    for cluster_id in range(n_clusters):
        cluster_data = cluster_df[cluster_df['Cluster'] == cluster_id]
        if len(cluster_data) > 0:
            most_common = cluster_data['Obesity_Class'].mode()[0]
            count = len(cluster_data)
            percentage = (count / len(cluster_df)) * 100
            print(f" - Cluster {cluster_id} (n={count}, {percentage:.1f}%): Most common = {most_common}")
    
    # Discussion
    print(f"\nDBSCAN with heurestic parameter selections gave a value of eps as {heuristic_eps:.5} and {heuristic_clusters} clusters which does not give us much information. Applied a multipler to attempt to reduce the value of eps to get more meaningful data. This reduction gave us an optimal eps of {optimal_eps:.5} and {n_clusters} with a silhouette score of {best_silhouette_float:.5}. Overall DBSCAN does not perform well on this dataset due to the large dimension ({d_original}) of this dataset.\n")    
   
    return {
        'best_eps': float(optimal_eps),
        'best_min_samples': int(min_samples),
        'best_n_clusters': int(n_clusters),
        'best_silhouette': float(best_silhouette_float),
        'n_noise': int(n_noise),
        'noise_ratio': float(noise_ratio),
        'calinski_harabasz': float(calinski_harabasz),
        'davies_bouldin': float(davies_bouldin),
        'heuristic_used': True,
        'optimal_eps_heuristic': float(optimal_eps),
        'min_samples_heuristic': int(min_samples),
        'dimensions_used': int(d_original)
    }