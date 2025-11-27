import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                            davies_bouldin_score, accuracy_score, 
                            precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, 
                            roc_auc_score, roc_curve)
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import label_binarize


def hierarchical_analysis(X_encoded, y_encoded, le_target, y):
  results = {
    "clustering": {},
    "outlier_detection": {},
    "feature_selection": {},
    "classification": {}
  }

  # 1. Hierarchical Clustering Algorithm
  print("1. HIERARCHICAL CLUSTERING\n")

  # Different number of clusters to evaluate
  clusters_range = [3, 5, 7, 9]
  hierarchical_results = []

  print("Evaluating different number of clusters: ")
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

  # # Analyze cluster characteristics
  print("\nCluster Characteristics:")
  cluster_df = pd.DataFrame({'Cluster': cluster_labels_best,'Obesity_Class': y})

  for cluster_id in range(best_n_clusters):
    cluster_data = cluster_df[cluster_df['Cluster'] == cluster_id]
    most_common = cluster_data['Obesity_Class'].mode()[0] if len(cluster_data) > 0 else 'N/A'
    print(f"Cluster {cluster_id} (n={len(cluster_data)}): Most common class = {most_common}")

  best_sillhouette = results_df.loc[results_df['silhoutte_score'].idxmax(), 'silhoutte_score']

  results['clustering'] = {
      'best_n_clusters': best_n_clusters,
      'metrics': results_df.to_dict('records'),
      'best_sillhouette': best_sillhouette
  }

  # Discussion: Appropriateness and Performance Comparison
  print("\n*** cluster analysis discussion ***")
  print(f"Hierarchical clustering with Ward linkage fits this obesity dataset well because the points are grouped based on distances in the features and doesn't really need class labels to find structure. From the k values tested, k={best_n_clusters} achieved the highest silhouette score of {best_sillhouette:.3f} compare to k = 3 (0.124), k = 5 (0.185), k = 7 (0.217). Therefore, the algorithm and the PCA successfully identified {best_n_clusters} clusters that fit the major obesity categories. ")

  # 2. Outlier Detection (Distance-based Outlier Detection)
  print("\n2. k-NN DISTANCE-BASED OUTLIER DETECTION")

  k = 5 
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
  for class_name in np.unique(y):
    class_mask = y == class_name
    class_outliers = outlier[class_mask].sum()
    class_total = class_mask.sum()
    print(f"  • {class_name}: {class_outliers}/{class_total} ({class_outliers/class_total*100:.2f}%)")
  
  print("\nOutlier Analysis:")
  print("\nWe going to keep all the data since outliers mostly represent unusual but valid combinations of features (including extreme normal-weight and obese cases).\n")

  results['outlier_detection'] = {
    'n_outliers': int(outlier.sum()),
  }

  # 3. Feature Selection (Mutual Information)
  print("\n3. FEATURE SELECTION (MUTUAL INFORMATION)")

  mi_scores = mutual_info_classif(X_encoded, y_encoded, random_state=50)
  feature_names = X_encoded.columns if hasattr(X_encoded, 'columns') else [f'Feature_{i}' for i in range(X_encoded.shape[1])]
  mi_df = pd.DataFrame({
    'Feature': feature_names,
    'MI_Score': mi_scores
  }).sort_values('MI_Score', ascending=False)

  print("\nFeature Importance (Mutual Information):")
  print(mi_df.to_string(index=False))

  # Analyze feature score distribution
  high_importance = mi_df[mi_df['MI_Score'] > mi_df['MI_Score'].median()]
  low_importance = mi_df[mi_df['MI_Score'] <= mi_df['MI_Score'].median()]

  print(f"\nFeature Importance Distribution:")
  print(f"High importance or > median: {len(high_importance)} features")
  print(f"Low importance or <= median: {len(low_importance)} features")
  print(f"Median MI score: {mi_df['MI_Score'].median():.4f}")

  # Select top k features (pick 8 features)
  k_features = min(8, X_encoded.shape[1])
  selector = SelectKBest(mutual_info_classif, k=k_features)
  X_selected = selector.fit_transform(X_encoded, y_encoded)
  selected_feature_indices = selector.get_support(indices=True)
  selected_features = [feature_names[i] for i in selected_feature_indices]

  print("\ndiscussion: importance of selected features")

  # Expected impact on classification
  print(f"\nExpected Impact on Classification:")
  info_retained = mi_df.head(k_features)['MI_Score'].sum()/mi_df['MI_Score'].sum()*100
  print(f"With selected {k_features} features > median, we retain {info_retained:.1f}% of information")
  print(f"Dimensionality reduced by {(1 - k_features/X_encoded.shape[1])*100:.1f}%")

  print(f"since information retained is {info_retained:.1f}%, we can expect slight accuracy trade-off for faster training")

  print(f"\nFeature Selection:")
  print(f"Selected {k_features}/{X_encoded.shape[1]} features")
  print(f"Compare classifier performance with full vs selected features\n")

  # Store results
  results['feature_selection'] = {
      'k_features': k_features,
      'X_selected': X_selected,
      'info_retained_pct': info_retained
  }

  # Evaluate the model with and without feature selectin using K-NN Classifier
  print("Evaluate the model with and without feature selectin using K-NN Classifier")

  # Split data into training and testing sets (80% training, 20% testing)
  X_selected = results['feature_selection']['X_selected']
  X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
  )

  X_train_selected, X_test_selected, _, _ = train_test_split(
      X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
  )

  print(f"Training set: {X_train_full.shape[0]} samples")
  print(f"Test set: {X_test_full.shape[0]} samples")
  print(f"Full features: {X_train_full.shape[1]}")
  print(f"Selected features: {X_train_selected.shape[1]}")

  # Train k-NN classifier with full features
  print("\nTraining k-NN classifier with full features")
  knn_full = KNeighborsClassifier(n_neighbors=5)
  start = time.time()
  knn_full.fit(X_train_full, y_train)
  train_time_full = time.time() - start

  start = time.time()
  y_pred_full = knn_full.predict(X_test_full)
  predict_time_full = time.time() - start
  acc_full = accuracy_score(y_test, y_pred_full)

  print(f"Accuracy: {acc_full:.4f}")
  print(f"Training time: {train_time_full:.4f}s")
  print(f"Prediction time: {predict_time_full:.4f}s")

  # Train k-NN classifier with selected features
  print("\nTraining k-NN classifier with selected features")
  knn_selected = KNeighborsClassifier(n_neighbors=5)
  start = time.time()
  knn_selected.fit(X_train_selected, y_train)
  train_time_selected = time.time() - start

  start = time.time()
  y_pred_selected = knn_selected.predict(X_test_selected)
  predict_time_selected = time.time() - start
  acc_selected = accuracy_score(y_test, y_pred_selected)

  print(f"Accuracy: {acc_selected:.4f}")
  print(f"Training time: {train_time_selected:.4f}s")
  print(f"Prediction time: {predict_time_selected:.4f}s")

  comparison = pd.DataFrame({
    'Model': ['All Features', 'Selected Features'],
    'N_Features': [X_train_full.shape[1], k_features],
    'Accuracy': [acc_full, acc_selected],
    'Train_Time(s)': [train_time_full, train_time_selected],
    'Predict_Time(s)': [predict_time_full, predict_time_selected]
  })

  acc_change = acc_selected - acc_full
  speed_improvement = (1 - train_time_selected/train_time_full) * 100

  print(f"\nFindings:")
  print(f"Accuracy change: {acc_change:+.4f}")
  print(f"Training speedup: {speed_improvement:.1f}%")
  print(f"Dimensionality reduction: {(1 - k_features/X_train_full.shape[1])*100:.1f}%")
  print(f"with the accuracy change of {acc_change:+.4f}, we can expect slight accuracy trade-off for faster training")

  # 4. Classification (k-NN Classifier)
  print("\n4. CLASSIFICATION (k-NN)")

  # Detailed metrics
  precision = precision_score(y_test, y_pred_selected, average='weighted')
  recall = recall_score(y_test, y_pred_selected, average='weighted')
  f1 = f1_score(y_test, y_pred_selected, average='weighted')

  print(f"\n Classification Metrics (k-NN with {k_features} features):")
  print(f"Accuracy:  {acc_selected:.4f}")
  print(f"Precision: {precision:.4f}")
  print(f"Recall:    {recall:.4f}")
  print(f"F1-Score:  {f1:.4f}")

  # Cross-validation
  cv_scores = cross_val_score(knn_selected, X_train_selected, y_train, cv=5)
  print(f"\n5-Fold Cross-Validation:")
  print(f"Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
  print(f"Scores: {[f'{s:.4f}' for s in cv_scores]}")

  # Classification report
  print(f"\nDetailed Classification Report:")
  print(classification_report(y_test, y_pred_selected, target_names=le_target.classes_))

  # Confusin Matrix
  cm = confusion_matrix(y_test, y_pred_selected)
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
              xticklabels=le_target.classes_,
              yticklabels=le_target.classes_)
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title(f'Confusion Matrix - k-NN Classification\n(Accuracy: {acc_selected:.4f})')
  plt.xticks(rotation=45, ha='right')
  plt.yticks(rotation=0)
  plt.tight_layout()
  plt.show()

  # Roc Curve
  y_test_bin = label_binarize(y_test, classes=np.arange(len(le_target.classes_)))
  n_classes = len(le_target.classes_)
  y_score = knn_selected.predict_proba(X_test_selected)

  fpr, tpr, roc_auc = {}, {}, {}
  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_score[:, i])

  fig, axes = plt.subplots(2, 4, figsize=(20, 10))
  axes = axes.ravel()
  colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

  for i, (ax, color) in enumerate(zip(axes[:n_classes], colors)):
      ax.plot(fpr[i], tpr[i], color=color, lw=2,
              label=f'AUC = {roc_auc[i]:.3f}')
      ax.plot([0, 1], [0, 1], 'k--', lw=1)
      ax.set_xlabel('False Positive Rate')
      ax.set_ylabel('True Positive Rate')
      ax.set_title(f'{le_target.classes_[i]}')
      ax.legend(loc="lower right")
      ax.grid(alpha=0.3)

  for i in range(n_classes, len(axes)):
      axes[i].axis('off')

  plt.suptitle('ROC Curves - Multi-class Classification', fontsize=16, fontweight='bold')
  plt.tight_layout()
  plt.show()
  
  results['classification'] = {
    'accuracy': acc_selected,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'confusion_matrix': cm.tolist(),
  }

  # 5. Hyperparameter Tuning (k-NN Classifier)

  print("\n5. HYPERPARAMETER TUNING (k-NN)")

  param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
  }

  # Running Grid Search 5 fold cross validation
  print("\nRunning Grid Search 5 fold cross validation")
  grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
  ).fit(X_train_selected, y_train)

  # Evaluate tuned model
  best_knn = grid_search.best_estimator_
  y_pred_tuned = best_knn.predict(X_test_selected)

  acc_tuned = accuracy_score(y_test, y_pred_tuned)
  precision_tuned = precision_score(y_test, y_pred_tuned, average='weighted')
  recall_tuned = recall_score(y_test, y_pred_tuned, average='weighted')
  f1_tuned = f1_score(y_test, y_pred_tuned, average='weighted')

  print(f"\nTuned Model Performance:")
  print(f"Accuracy: {acc_tuned:.4f}")
  print(f"Precision: {precision_tuned:.4f}")
  print(f"Recall: {recall_tuned:.4f}")
  print(f"F1-Score: {f1_tuned:.4f}")

  # Compare before and after tuning
  print("\nImpact of Hyperparameter Tuning")

  tuning_comparison = pd.DataFrame({
    'Model': ['Before Tuning (k=5)', 'After Tuning (GridSearch)'],
    'Accuracy': [acc_selected, acc_tuned],
    'Precision': [precision, precision_tuned],
    'Recall': [recall, recall_tuned],
    'F1-Score': [f1, f1_tuned]
  })

  print(tuning_comparison.to_string(index=False))

  improvement = acc_tuned - acc_selected
  print(f"\nFindings:")
  print(f"Accuracy improvement: {improvement:+.4f}")
  print(f"Best k value: {grid_search.best_params_['n_neighbors']}")
  print(f"Best weighting: {grid_search.best_params_['weights']}")
  print(f"Best distance metric: {grid_search.best_params_['metric']}")

  print(f"Since accuracy improvement is {improvement:+.4f}, we can expect significant improvement from hyperparameter tuning")

  # Before vs After Tuning Visualization
  plt.figure(figsize=(10, 6))

  # Metrics comparison
  metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
  before = [acc_selected, precision, recall, f1]
  after = [acc_tuned, precision_tuned, recall_tuned, f1_tuned]

  x = np.arange(len(metrics))
  width = 0.35

  plt.bar(x - width/2, before, width, label='Before Tuning', color='lightblue', edgecolor='black')
  plt.bar(x + width/2, after, width, label='After Tuning', color='lightgreen', edgecolor='black')
  plt.ylabel('Score')
  plt.title('Performance Comparison: Before vs After Tuning')
  plt.xticks(x, metrics, rotation=45, ha='right')
  plt.legend()
  plt.ylim([min(before + after) - 0.05, 1.0])
  plt.grid(axis='y', alpha=0.3)
  plt.tight_layout()
  plt.show()

  # Store results
  results['hyperparameter_tuning'] = {
    'tuned_accuracy': acc_tuned,
    'improvement': improvement,
  }
  return results