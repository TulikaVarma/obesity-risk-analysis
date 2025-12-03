# K-NN classification with metrics including cross-validation,
# confusion matrix, classification report, and ROC curve

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize

def knn_classification(train_data, test_data, feature_selection):
  # Reuse data from mutual_information
  X_train_selected = feature_selection['X_train_selected']
  X_test_selected = feature_selection['X_test_selected']
  y_train = feature_selection['y_train']
  y_test = feature_selection['y_test']
  le_target = feature_selection['le_target']
  k_features = feature_selection['k_features']

  # 4. Classification (k-NN Classifier)
  print("\n==== KNN Classification ====")

  # Train K-NN classifier using k = 5
  knn = KNeighborsClassifier(n_neighbors=5)
  knn.fit(X_train_selected, y_train)

  # Predictions
  y_pred = knn.predict(X_test_selected)

  # Detailed metrics
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average='weighted')
  recall = recall_score(y_test, y_pred, average='weighted')
  f1 = f1_score(y_test, y_pred, average='weighted')

  print(f"\n Classification Metrics (k-NN with {k_features} features):")
  print(f"Accuracy:  {accuracy:.4f}")
  print(f"Precision: {precision:.4f}")
  print(f"Recall:    {recall:.4f}")
  print(f"F1-Score:  {f1:.4f}")

  # Cross-validation
  cv_scores = cross_val_score(knn, X_train_selected, y_train, cv=5)
  print(f"\n5-Fold Cross-Validation:")
  print(f"Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
  print(f"Scores: {[f'{s:.4f}' for s in cv_scores]}")

  # Classification report
  print(f"\nDetailed Classification Report:")
  print(classification_report(y_test, y_pred, target_names=le_target.classes_))

  # Confusin Matrix
  cm = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
              xticklabels=le_target.classes_,
              yticklabels=le_target.classes_)
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title(f'Confusion Matrix - k-NN Classification\n(Accuracy: {accuracy:.4f})')
  plt.xticks(rotation=45, ha='right')
  plt.yticks(rotation=0)
  plt.tight_layout()
  plt.show()

    # Roc Curve
  y_test_bin = label_binarize(y_test, classes=np.arange(len(le_target.classes_)))
  n_classes = len(le_target.classes_)
  y_score = knn.predict_proba(X_test_selected)

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

  roc_auc_avg = np.mean(list(roc_auc.values()))
  
  return {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'roc_auc_avg': roc_auc_avg,
    'knn_model': knn,
    'X_train_selected': X_train_selected,
    'X_test_selected': X_test_selected,
    'y_train': y_train,
    'y_test': y_test,
    'le_target': le_target
  }