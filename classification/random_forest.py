import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def random_forest_classification(train_data, valid_data, test_data, feature_selection):
    X_train_selected = feature_selection["X_train_selected"]
    X_test_selected = feature_selection["X_test_selected"]
    y_train = feature_selection["y_train"]
    y_test = feature_selection["y_test"]
    le_target = feature_selection["le_target"]
    selected_features = feature_selection.get(
        "selected_features", [f"Feature_{i}" for i in range(X_train_selected.shape[1])]
    )
    k_features = feature_selection["k_features"]

    print("\n==== Random Forest Classification ====")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_selected, y_train)

    # Predictions
    y_pred = rf.predict(X_test_selected)
    y_score = rf.predict_proba(X_test_selected)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\nClassification Metrics (Random Forest with {k_features} features):")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Cross-validation on training set
    cv_scores = cross_val_score(rf, X_train_selected, y_train, cv=5)
    print(f"\n5-Fold Cross-Validation:")
    print(f"Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"Scores: {[f'{s:.4f}' for s in cv_scores]}")

    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=le_target.classes_,
        yticklabels=le_target.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - Random Forest\n(Accuracy: {accuracy:.4f})")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # ROC Curves
    y_test_bin = label_binarize(y_test, classes=np.arange(len(le_target.classes_)))
    n_classes = y_test_bin.shape[1]

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_score[:, i])

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    colors = plt.cm.Dark2(np.linspace(0, 1, n_classes))

    for i, (ax, color) in enumerate(zip(axes[:n_classes], colors)):
        ax.plot(fpr[i], tpr[i], color=color, lw=2, label=f"AUC = {roc_auc[i]:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{le_target.classes_[i]}")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

    for i in range(n_classes, len(axes)):
        axes[i].axis("off")

    plt.suptitle(
        "ROC Curves - Random Forest Multi-class Classification",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    roc_auc_avg = np.mean(list(roc_auc.values()))

    # Feature importance
    importance_df = pd.DataFrame(
        {"Feature": selected_features, "Importance": rf.feature_importances_}
    ).sort_values("Importance", ascending=False)

    print(f"\n--- Feature Importance (Top 10) ---")
    print(importance_df.head(10).to_string(index=False))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "roc_auc_avg": roc_auc_avg,
        "rf_model": rf,
        "X_train_selected": X_train_selected,
        "X_test_selected": X_test_selected,
        "y_train": y_train,
        "y_test": y_test,
        "le_target": le_target,
        "confusion_matrix": cm,
        "feature_importance": importance_df,
    }


class RandomForest:

    def __init__(
        self,
        n_trees: int = 10,
        max_depth=None,
        min_samples_split: int = 2,
        max_features=None,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = X.columns if hasattr(X, "columns") else None
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        data = X.loc[:, self.feature_names_] if self.feature_names_ is not None else X
        return self.model.predict(data)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == np.array(y)))
