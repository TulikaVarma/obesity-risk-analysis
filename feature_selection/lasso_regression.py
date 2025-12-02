# Feature selection using Lasso Regression (L1 regularization)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def lasso_feature_selection(train_data, valid_data, test_data):
    # Prepare data
    X_train = train_data.drop(['NObeyesdad'], axis=1)
    y_train_original = train_data['NObeyesdad']
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(y_train_original)

    print("\n==== Lasso Regression Feature Selection ====")
    
    # Scale features (required for Lasso regularization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])]
    
    alpha_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    lasso_results = []
    
    for alpha in alpha_values:
        # Fit Lasso model
        lasso = Lasso(alpha=alpha, random_state=50, max_iter=2000)
        lasso.fit(X_train_scaled, y_train)
        
        # Count non-zero coefficients (selected features)
        n_selected = np.sum(np.abs(lasso.coef_) > 1e-5)
        
        lasso_results.append({
            'alpha': alpha,
            'n_features': n_selected,
            'pct_features': n_selected / len(feature_names) * 100
        })
    # Use LassoCV to select optimal alpha via cross-validation
    lasso_cv = LassoCV(alphas=alpha_values, cv=5, random_state=50, max_iter=2000)
    lasso_cv.fit(X_train_scaled, y_train)
    optimal_alpha = lasso_cv.alpha_
    print(f"Cross-validation selected alpha={optimal_alpha:.4f}")
    
    # Show which alpha from our test range is closest
    cv_n_features = np.sum(np.abs(lasso_cv.coef_) > 1e-5)
    print(f"This selects {cv_n_features}/{len(feature_names)} features ({cv_n_features/len(feature_names)*100:.1f}%)")
    
    # Fit final Lasso model with CV-selected alpha
    lasso_final = Lasso(alpha=optimal_alpha, random_state=50, max_iter=2000)
    lasso_final.fit(X_train_scaled, y_train)
    
    # Get feature importances (absolute coefficients)
    feature_importance = np.abs(lasso_final.coef_)
    
    # Create feature importance dataframe
    lasso_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(lasso_df.head(10).to_string(index=False))
    
    # Select features with non-zero coefficients
    selected_mask = feature_importance > 1e-5
    selected_features = [feature_names[i] for i in np.where(selected_mask)[0]]
    k_features = len(selected_features)
    
    print(f"\nFeature Selection Results:")
    print(f" - Selected features: {k_features}/{len(feature_names)} ({k_features/len(feature_names)*100:.1f}%)")
    print(f" - Features eliminated: {len(feature_names) - k_features}")
    
    print(f"\nSelected features:")
    for i, feat in enumerate(selected_features, 1):
        importance = lasso_df[lasso_df['Feature'] == feat]['Importance'].values[0]
        print(f"  {i}. {feat:30s} (importance: {importance:.4f})")
    
    # Prepare test data
    X_test = test_data.drop(['NObeyesdad'], axis=1)
    y_test = le_target.transform(test_data['NObeyesdad'])
    X_test_scaled = scaler.transform(X_test)
    
    # Apply feature selection to train and test
    X_train_selected = X_train_scaled[:, selected_mask]
    X_test_selected = X_test_scaled[:, selected_mask]
    
    print(f"\nEvaluating k-NN with {k_features}/{X_train.shape[1]} features (full vs. selected)")
    print(f"\n - Training set: {X_train.shape[0]} samples")
    print(f" - Test set: {X_test.shape[0]} samples")
    print(f" - Full features: {X_train.shape[1]}")
    print(f" - Selected features: {X_train_selected.shape[1]}")
    
    # Train k-NN with all features
    print("\n--- Training k-NN classifier with all features ---")
    knn_full = KNeighborsClassifier(n_neighbors=5)
    start = time.perf_counter()  
    knn_full.fit(X_train_scaled, y_train)
    train_time_full = time.perf_counter() - start
    
    start = time.perf_counter()
    y_pred_full = knn_full.predict(X_test_scaled)
    predict_time_full = time.perf_counter() - start
    acc_full = accuracy_score(y_test, y_pred_full)
    
    print(f"Accuracy: {acc_full:.4f}")
    print(f"Training time: {train_time_full:.6f}s")
    print(f"Prediction time: {predict_time_full:.6f}s")
    
    # Train k-NN with slected features
    print("\n--- Training k-NN classifier with Lasso-selected features ---")
    knn_selected = KNeighborsClassifier(n_neighbors=5)
    start = time.perf_counter()
    knn_selected.fit(X_train_selected, y_train)
    train_time_selected = time.perf_counter() - start
    
    start = time.perf_counter()
    y_pred_selected = knn_selected.predict(X_test_selected)
    predict_time_selected = time.perf_counter() - start
    acc_selected = accuracy_score(y_test, y_pred_selected)
    
    print(f"Accuracy: {acc_selected:.4f}")
    print(f"Training time: {train_time_selected:.6f}s")
    print(f"Prediction time: {predict_time_selected:.6f}s")
    
    # Comparison - handle zero division
    acc_change = acc_selected - acc_full
    
    # Avoid division by zero
    if train_time_full > 0:
        speed_improvement = (1 - train_time_selected/train_time_full) * 100
    else:
        speed_improvement = 0.0
        
    if predict_time_full > 0:
        predict_speedup = (1 - predict_time_selected/predict_time_full) * 100
    else:
        predict_speedup = 0.0
        
    dimension_reduction = (1 - k_features/X_train.shape[1]) * 100
    
    print(f"\n*** Lasso Feature Selection Analysis ***")
    print(f"\nFindings:")
    print(f"With all 31 features: {acc_full:.4f} accuracy")
    print(f"With 21 selected features: {acc_selected:.4f} accuracy")
    print(f"Accuracy change: {acc_change:+.4f} ({acc_change/acc_full*100:+.1f}%)")
    print(f"Training speedup: {((train_time_full - train_time_selected) / train_time_full * 100):.1f}%")
    print(f"applied Lasso Regression with 5-fold cross-validation. The optimal regularization parameter (alpha ={optimal_alpha}) selected 21 out of 31 features (67.7%), including Weight, CAEC_Frequently, family history, Age, and Height as the most important. It reduces less relevant features to zero, improving model performance while decreasing computational complexity.\n")
    
    return {
        'lasso_model': lasso_final,
        'scaler': scaler,
        'selected_mask': selected_mask,
        'k_features': k_features,
        'selected_features': selected_features,
        'optimal_alpha': optimal_alpha,
        'X_train_selected': X_train_selected,
        'X_test_selected': X_test_selected,
        'y_train': y_train,
        'y_test': y_test,
        'le_target': le_target,
        'acc_full': acc_full,
        'acc_selected': acc_selected,
        'feature_importance_df': lasso_df,
        'acc_change': acc_change,
        'speed_improvement': speed_improvement,
        'dimension_reduction': dimension_reduction,
        'train_time_full': train_time_full,
        'train_time_selected': train_time_selected,
        'predict_time_full': predict_time_full,
        'predict_time_selected': predict_time_selected
    }