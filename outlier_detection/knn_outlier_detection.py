# KNN distance based outlier detection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

def knn_outlier_detection(data, X_pca_2d=None, k=5):
  # Prepare data
  X_encoded = data.drop(['NObeyesdad'], axis=1)
  y_original = data['NObeyesdad']
  
  # 2. Outlier Detection (Distance-based Outlier Detection)
  print("\n==== KNN Distance-Based Outlier Detection ====")
  print(f"Using k = {k} for KNN")

  # Fit KNN model for outlier detection
  knn = NearestNeighbors(n_neighbors=k).fit(X_encoded)

  # Get distances of nearest neighbors
  distances, _ = knn.kneighbors(X_encoded)

  # Calculate outlier scores using avg distance to k nearest neighbors, using the 95th percentile as the threshold
  scores = distances.mean(axis=1)
  threshold = np.percentile(scores, 95)
  outlier = scores > threshold
  
  print(f"\nOutlier Detection Results:")
  print(f"  • Threshold (95th percentile): {threshold:.4f}")
  print(f"  • Outliers detected: {outlier.sum()} ({outlier.sum()/len(outlier)*100:.2f}%)")
  print(f"  • Normal samples: {(~outlier).sum()}")

  # Visualize outliers
  print("\nVisualizing outliers")    
  # Plot: Outliers on PCA scatter plot
  plt.figure(figsize=(8, 6))
  plt.scatter(X_pca_2d[~outlier, 0], X_pca_2d[~outlier, 1], color='blue', alpha=0.5, label='Normal', s=20)
  plt.scatter(X_pca_2d[outlier, 0], X_pca_2d[outlier, 1], color='red', alpha=0.8, label='Outlier', s=50, marker='x', linewidths=2)
  plt.xlabel('PC1')
  plt.ylabel('PC2')
  plt.title('Outlier Detection - PCA Visualization')
  plt.legend()
  plt.grid(alpha=0.3)
  plt.tight_layout()
  plt.show()

  print("\nOutlier Distribution by Obesity Class:")
  for class_name in np.unique(y_original):
    class_mask = y_original == class_name
    class_outliers = outlier[class_mask].sum()
    class_total = class_mask.sum()
    print(f"  • {class_name}: {class_outliers}/{class_total} ({class_outliers/class_total*100:.2f}%)")
  
  print("\nOutlier Analysis:")
  print("\nWe going to keep all the data since outliers mostly represent unusual but valid combinations of features (including extreme normal-weight and obese cases).\n")

  return {
    'n_outliers': int(outlier.sum()),
    'percentage': outlier.sum() / len(outlier) * 100,
    'threshold': threshold,
  }