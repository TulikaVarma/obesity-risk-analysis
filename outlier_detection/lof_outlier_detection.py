# Local Outlier Factor (LOF) based outlier detection

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor


def lof_outlier_detection(data, X_pca_2d=None, n_neighbors=20, contamination=0.05):
  # Prepare data
  X_encoded = data.drop(['NObeyesdad'], axis=1)
  y_original = data['NObeyesdad']

  print("\n==== LOF Outlier Detection ====")
  print(f"Using n_neighbors={n_neighbors}, contamination={contamination}")

  lof = LocalOutlierFactor(
    n_neighbors=n_neighbors,
    contamination=contamination,
    novelty=False,
    metric="minkowski",
  )
  preds = lof.fit_predict(X_encoded)
  scores = -lof.negative_outlier_factor_

  outlier = preds == -1
  threshold = np.percentile(scores, 100 * (1 - contamination))

  print(f"\nOutlier Detection Results:")
  print(f"  • Score threshold (approx.): {threshold:.4f}")
  print(f"  • Outliers detected: {outlier.sum()} ({outlier.sum()/len(outlier)*100:.2f}%)")
  print(f"  • Normal samples: {(~outlier).sum()}")

  # Visualize outliers on provided PCA projection
  if X_pca_2d is not None:
    print("\nVisualizing outliers")
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca_2d[~outlier, 0], X_pca_2d[~outlier, 1], color='blue', alpha=0.5, label='Normal', s=20)
    plt.scatter(X_pca_2d[outlier, 0], X_pca_2d[outlier, 1], color='red', alpha=0.8, label='Outlier', s=50, marker='x', linewidths=2)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('LOF Outlier Detection - PCA Visualization')
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

  return {
    'n_outliers': int(outlier.sum()),
    'percentage': outlier.sum() / len(outlier) * 100,
    'contamination': contamination,
    'threshold_estimate': float(threshold),
  }
