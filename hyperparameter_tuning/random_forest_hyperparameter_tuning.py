import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def random_forest_hyperparameter_tuning(rf_results):
  X_train = rf_results['X_train_selected']
  X_test = rf_results['X_test_selected']
  y_train = rf_results['y_train']
  y_test = rf_results['y_test']
  acc_before = rf_results['accuracy']
  precision_before = rf_results['precision']
  recall_before = rf_results['recall']
  f1_before = rf_results['f1_score']

  print("\n ==== Hyperparameter Tuning for Random Forest ====")

  param_grid = {
    'n_estimators': [150, 300, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None],
  }

  grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
  ).fit(X_train, y_train)

  best_rf = grid_search.best_estimator_
  y_pred_tuned = best_rf.predict(X_test)

  acc_tuned = accuracy_score(y_test, y_pred_tuned)
  precision_tuned = precision_score(y_test, y_pred_tuned, average='weighted')
  recall_tuned = recall_score(y_test, y_pred_tuned, average='weighted')
  f1_tuned = f1_score(y_test, y_pred_tuned, average='weighted')

  print(f"\nTuned Model Performance:")
  print(f"Accuracy:  {acc_tuned:.4f}")
  print(f"Precision: {precision_tuned:.4f}")
  print(f"Recall:    {recall_tuned:.4f}")
  print(f"F1-Score:  {f1_tuned:.4f}")

  tuning_comparison = pd.DataFrame({
    'Model': ['Before Tuning', 'After Tuning (GridSearch)'],
    'Accuracy': [acc_before, acc_tuned],
    'Precision': [precision_before, precision_tuned],
    'Recall': [recall_before, recall_tuned],
    'F1-Score': [f1_before, f1_tuned]
  })

  print("\nImpact of Hyperparameter Tuning")
  print(tuning_comparison.to_string(index=False))

  improvement = acc_tuned - acc_before
  print(f"\nFindings:")
  print(f"Accuracy improvement: {improvement:+.4f}")
  print(f"Best params: {grid_search.best_params_}")

  return {
    'best_params': grid_search.best_params_,
    'best_cv_score': grid_search.best_score_,
    'tuned_accuracy': acc_tuned,
    'tuned_precision': precision_tuned,
    'tuned_recall': recall_tuned,
    'tuned_f1': f1_tuned,
    'improvement': improvement,
  }
