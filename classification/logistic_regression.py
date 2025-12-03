# Logistic Regression classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize

def logistic_regression_classification(train_data, test_data, feature_selection):
    # Reuse data from Lasso feature selection
    X_train_selected = feature_selection['X_train_selected']
    X_test_selected = feature_selection['X_test_selected']
    y_train = feature_selection['y_train']
    y_test = feature_selection['y_test']
    le_target = feature_selection['le_target']
    k_features = feature_selection['k_features']
    selected_features = feature_selection['selected_features']
    
    print("\n==== Logistic Regression Classification ====\n")
    
    # Train Logistic Regression classifier
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    lr.fit(X_train_selected, y_train)
    
    # Predictions on test
    y_pred = lr.predict(X_test_selected)
    y_score = lr.predict_proba(X_test_selected)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Classification Metrics (Logistic Regression with {k_features} features):")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(lr, X_train_selected, y_train, cv=5)
    print(f"\n5-Fold Cross-Validation:")
    print(f"Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"Scores: {[f'{s:.4f}' for s in cv_scores]}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_, zero_division=0))
    
    # ROC-AUC Analysis
    y_test_bin = label_binarize(y_test, classes=np.arange(len(le_target.classes_)))
    n_classes = len(le_target.classes_)
    
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_score[:, i])
    
    roc_auc_avg = np.mean(list(roc_auc.values()))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Logistic Regression Classification\n(Accuracy: {accuracy:.4f})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # ROC Curves
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for i, (ax, color) in enumerate(zip(axes[:n_classes], colors)):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'AUC = {roc_auc[i]:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'{le_target.classes_[i]}', fontsize=10, fontweight='bold', pad=10)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
    
    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('ROC Curves - Multi-class Classification (Logistic Regression)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    return {
        'model': lr,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores,
        'roc_auc_avg': roc_auc_avg,
        'roc_auc_per_class': roc_auc,
        'X_train_selected': X_train_selected,
        'X_test_selected': X_test_selected,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_score': y_score,
        'le_target': le_target,
        'confusion_matrix': cm,
        'selected_features': selected_features,
        'k_features': k_features,
        'n_classes': n_classes,
        'fpr': fpr,
        'tpr': tpr
    }