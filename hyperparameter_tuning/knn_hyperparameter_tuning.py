# K-NN hyperparameter tuning using GridSearchCV

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def knn_hyperparameter_tuning(knn_classification_results):
  # Get Data
  X_train_selected = knn_classification_results['X_train_selected']
  X_test_selected = knn_classification_results['X_test_selected']
  y_train = knn_classification_results['y_train']
  y_test = knn_classification_results['y_test']
  
  # Get baseline metrics from Section 4
  acc_before = knn_classification_results['accuracy']
  precision_before = knn_classification_results['precision']
  recall_before = knn_classification_results['recall']
  f1_before = knn_classification_results['f1_score']

  print("\n ==== Hyperparameter Tuning for KNN ====")

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
    'Accuracy': [acc_before, acc_tuned],
    'Precision': [precision_before, precision_tuned],
    'Recall': [recall_before, recall_tuned],
    'F1-Score': [f1_before, f1_tuned]
  })

  print(tuning_comparison.to_string(index=False))

  improvement = acc_tuned - acc_before
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
  before = [acc_before, precision_before, recall_before, f1_before]
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

  return {
    'best_params': grid_search.best_params_,
    'best_cv_score': grid_search.best_score_,
    'tuned_accuracy': acc_tuned,
    'tuned_precision': precision_tuned,
    'tuned_recall': recall_tuned,
    'tuned_f1': f1_tuned,
    'improvement': improvement,
  }