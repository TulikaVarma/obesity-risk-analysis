# Feature selection using mutual information, select the top features based on the mutual information score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def mutual_information(train_data, valid_data, test_data):
  # Prepare data
  X_encoded = train_data.drop(['NObeyesdad'], axis=1)
  y_original = train_data['NObeyesdad']
  le_target = LabelEncoder()
  y_encoded_full = le_target.fit_transform(y_original)

  # 3. Feature Selection (Mutual Information)
  print("\n ==== Mutual Information Feature Selection ====")

  mi_scores = mutual_info_classif(X_encoded, y_encoded_full, random_state=50)
  feature_names = X_encoded.columns if hasattr(X_encoded, 'columns') else [f'Feature_{i}' for i in range(X_encoded.shape[1])]
  mi_df = pd.DataFrame({
    'Feature': feature_names,
    'MI_Score': mi_scores
  }).sort_values('MI_Score', ascending=False)

  print("\nTop 10 Most Important Features:")
  print(mi_df.head(10).to_string(index=False))

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
  X_selected = selector.fit_transform(X_encoded, y_encoded_full)
  selected_feature_indices = selector.get_support(indices=True)
  selected_features = [feature_names[i] for i in selected_feature_indices]
  info_retained = mi_df.head(k_features)['MI_Score'].sum() / mi_df['MI_Score'].sum() * 100

  print(f"\nEvaluating k-NN with {k_features}/{X_encoded.shape[1]} features (full vs. selected)\n")

  # Prepare train/test data
  X_train = train_data.drop(['NObeyesdad'], axis=1)
  y_train = le_target.transform(train_data['NObeyesdad'])
  X_test = test_data.drop(['NObeyesdad'], axis=1)
  y_test = le_target.transform(test_data['NObeyesdad'])
  
  # Scale the data for k-NN (k-NN is distance-based and benefits from scaling)
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  
  # Apply feature selection to scaled data
  X_train_selected = selector.transform(X_train_scaled)
  X_test_selected = selector.transform(X_test_scaled)
  
  print(f"\nTraining set: {X_train.shape[0]} samples")
  print(f"Test set: {X_test.shape[0]} samples")
  print(f"Full features: {X_train.shape[1]}")
  print(f"Selected features: {X_train_selected.shape[1]}")
  
  # Train k-NN with full scaled features
  print("\nTraining k-NN classifier with full features...")
  knn_full = KNeighborsClassifier(n_neighbors=5)
  start = time.time()
  knn_full.fit(X_train_scaled, y_train)
  train_time_full = time.time() - start
  
  start = time.time()
  y_pred_full = knn_full.predict(X_test_scaled)
  predict_time_full = time.time() - start
  acc_full = accuracy_score(y_test, y_pred_full)
  
  print(f"Accuracy: {acc_full:.4f}")
  print(f"Training time: {train_time_full:.4f}s")
  print(f"Prediction time: {predict_time_full:.4f}s")
  
  # Train k-NN with selected features
  print("\nTraining k-NN classifier with selected features...")
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
  
  # Comparison
  acc_change = acc_selected - acc_full
  speed_improvement = (1 - train_time_selected/train_time_full) * 100

  print(f"\nFindings:")
  print(f"Accuracy change: {acc_change:+.4f}")
  print(f"Training speedup: {speed_improvement:.1f}%")
  print(f"Dimensionality reduction: {(1 - k_features/X_train.shape[1])*100:.1f}%")
  print(f"With the accuracy change of {acc_change:+.4f}, we can expect slight accuracy trade-off for faster training\n")

  return {
    'selector': selector,
    'k_features': k_features,
    'selected_features': selected_features,
    'info_retained_pct': info_retained,
    'X_train_selected': X_train_selected,
    'X_test_selected': X_test_selected,
    'y_train': y_train,
    'y_test': y_test,
    'le_target': le_target,
    'acc_full': acc_full,
    'acc_selected': acc_selected,
  }