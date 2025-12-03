import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from classification.random_forest import random_forest_classification
from clustering.hierarchical_clustering import hierarchical_clustering
from clustering.clarans import clarans_clustering
from outlier_detection.knn_outlier_detection import knn_outlier_detection
from outlier_detection.lof_outlier_detection import lof_outlier_detection
from classification.knn_classification import knn_classification
from hyperparameter_tuning.knn_hyperparameter_tuning import knn_hyperparameter_tuning
from hyperparameter_tuning.random_forest_hyperparameter_tuning import random_forest_hyperparameter_tuning
from hyperparameter_tuning.logistic_regression_hyperparameter_tuning import logistic_regression_hyperparameter_tuning
from clustering.dbscan_clustering import dbscan_clustering
from outlier_detection.probabilistic_outlier_detection import probabilistic_outlier_detection
from feature_selection.lasso_regression import lasso_feature_selection
from classification.logistic_regression import logistic_regression_classification
from sklearn.preprocessing import StandardScaler

def load_and_split_data():
  # Load and split data consistently (60/20/20)
  data = pd.read_csv("cleaned_data.csv")
  data = data.sample(frac=1, random_state=30).reset_index(drop=True)
  
  train_size = int(0.8 * len(data))
  # valid_size = int(0.2 * len(data))
  
  train_data = data.iloc[:train_size]
  # valid_data = data.iloc[train_size:train_size + valid_size]
  test_data = data.iloc[train_size:]
  
  print(f"Data Split:")
  print(f"Total Samples: {len(data)}")
  print(f"Training Samples: {len(train_data)} (60%)")
  # print(f"Validation Samples: {len(valid_data)} (20%)")
  print(f"Test Samples: {len(test_data)} (20%)")
  
  return data, train_data, test_data

def clustering_analysis(train_data):
  # Clustering Analysis
  print("1. CLUSTERING ANALYSIS")

  X = train_data.drop(['NObeyesdad'], axis=1)
  y = train_data['NObeyesdad']
  
  # Standardize and apply PCA
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  pca = PCA(n_components=15)
  X_pca = pca.fit_transform(X_scaled)

  print(f"Original features: {X.shape[1]} to PCA components: {pca.n_components_}")
  print(f"Variance explained: {pca.explained_variance_ratio_.sum():.1%}\n")

  hierarchical_results = hierarchical_clustering(X_pca, y)
  clarans_results = clarans_clustering(X_pca, y)
  dbscan_results = dbscan_clustering(X_pca, y)
  clustering_results = {
    'hierarchical': hierarchical_results,
    'clarans': clarans_results,
    'dbscan': dbscan_results,
  }

  # Compare all three clustering results and discussion
  # Compare using silhouette, calinski, davies
  print("*** Clustering comparison summary***")
  print(f"{'Algorithm':<15} {'k':<5} {'Silhouette':<12} {'Calinski-H':<12} {'Davies-B':<12}")
  print("-"*60)

  # Hierarchical
  print(f"{'Hierarchical':<15} {hierarchical_results['best_n_clusters']:<5} {hierarchical_results['best_sillhouette']:<12.3f} {hierarchical_results['calinski_harabasz']:<12.1f} {hierarchical_results['davies_bouldin']:<12.3f}")

  # CLARANS
  print(f"{'CLARANS':<15} {clarans_results['best_n_clusters']:<5} {clarans_results['best_silhouette']:<12.3f} {clarans_results['calinski_harabasz']:<12.1f} {clarans_results['davies_bouldin']:<12.3f}")

  # DBSCAN
  print(f"{'DBSCAN':<15} {dbscan_results['best_n_clusters']:<5} {dbscan_results['best_silhouette']:<12.3f} {dbscan_results['calinski_harabasz']:<12.1f} {dbscan_results['davies_bouldin']:<12.3f} ")

  print("-"*60)

  # Summarize
  print(f"\nAll algorithms evaluated on same 15 PCA components (86.4% variance) with DBSCAN achieved the highest silhouette score ({dbscan_results['best_silhouette']:.3f}) and Davies-Bouldin Index ({dbscan_results['davies_bouldin']:.3f}). While Hierarchical are better in Calinski-Harabasz Index ({hierarchical_results['calinski_harabasz']:.1f})\n")

  return clustering_results

def outlier_detection(train_data):
  # Outlier Detection
  print("2. OUTLIER DETECTION")

  # Compute PCA once for all outlier detectionvisualizations
  X_encoded = train_data.drop(['NObeyesdad'], axis=1)
  pca = PCA(n_components=2)
  X_pca_2d = pca.fit_transform(X_encoded)

  knn_outlier_results = knn_outlier_detection(train_data, X_pca_2d=X_pca_2d)
  probabilistic_outlier_results = probabilistic_outlier_detection(train_data, X_pca_2d=X_pca_2d)
  lof_outlier_results = lof_outlier_detection(train_data, X_pca_2d=X_pca_2d)

  outlier_detection_results = {
    'knn': knn_outlier_results,
    'probabilistic': probabilistic_outlier_results,
    'lof': lof_outlier_results,
  }

  # Analyze all three outlier detection results and discussion
  print("\n*** Outlier Detection Comparison Summary ***")
  print(f"{'Method':<20} {'Outliers':<12} {'Percentage':<12} {'Threshold':<15}")
  print("-"*60)
  
  print(f"{'k-NN Distance':<20} {knn_outlier_results['n_outliers']:<12} {knn_outlier_results['percentage']:<12.2f} {knn_outlier_results['threshold']:<15.4f}")
  print(f"{'Probabilistic (GMM)':<20} {probabilistic_outlier_results['n_outliers']:<12} {probabilistic_outlier_results['percentage']:<12.2f} {probabilistic_outlier_results['threshold']:<15.4f}")
  print(f"{'LOF':<20} {lof_outlier_results['n_outliers']:<12} {lof_outlier_results['percentage']:<12.2f} {lof_outlier_results.get('threshold_estimate', 0):<15.4f}")
  
  print("-"*60)
  
  print(f"All three methods detected similar outlier counts ({knn_outlier_results['n_outliers']}-{lof_outlier_results['n_outliers']} outliers) which is about 5%, showing consistency across different detection approaches.\n")
  print("Normal weight class shows highest outlier rate (15-21% across methods) showing variability, while Obesity_Type_III shows lowest outlier rate (0-5%) showing consitency in obesity cases.")
  
  print("we decide to keep all the data since outliers mostly represent unusual but valid combinations of features (including extreme normal-weight and obese cases).")

  return outlier_detection_results

def feature_selection(train_data, test_data):
  # Feature Selection
  print("3. FEATURE SELECTION") 

  lasso_results = lasso_feature_selection(train_data, test_data)
  feature_selection_results = {
    'lasso': lasso_results,
  }
  print("-"*60 + "\n")

  return feature_selection_results

def classification(train_data, test_data, feature_selection_results):
  # Classification
  print("4. CLASSIFICATION")

  knn_results = knn_classification(train_data, test_data, feature_selection_results['lasso'])
  lr_results = logistic_regression_classification(train_data, test_data, feature_selection_results['lasso']) 
  rf_results = random_forest_classification(train_data, test_data, feature_selection_results['lasso'])

  classification_results = {
    'knn': knn_results,
    'logistic_regression': lr_results,
    'random_forest': rf_results,
  }

  # Classification summary
  print("\n*** CLASSIFICATION COMPARISON SUMMARY ***")
  print(f"{'Classifier':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'CV Mean':<12}")
  print("-"*80)

  print(f"{'k-NN':<20} {knn_results['accuracy']:<12.4f} {knn_results['precision']:<12.4f} {knn_results['recall']:<12.4f} {knn_results['f1_score']:<12.4f} {knn_results['cv_mean']:<12.4f}")
  print(f"{'Logistic Regression':<20} {lr_results['accuracy']:<12.4f} {lr_results['precision']:<12.4f} {lr_results['recall']:<12.4f} {lr_results['f1_score']:<12.4f} {lr_results['cv_mean']:<12.4f}")
  print(f"{'Random Forest':<20} {rf_results['accuracy']:<12.4f} {rf_results['precision']:<12.4f} {rf_results['recall']:<12.4f} {rf_results['f1_score']:<12.4f} {rf_results['cv_mean']:<12.4f}")
  
  print("-"*80 + "\n")
  
  return classification_results


def hyperparameter_tuning(classification_results):
  # Hyperparameter Tuning
  print("5. HYPERPARAMETER TUNING")

  knn_tuned = knn_hyperparameter_tuning(classification_results['knn'])
  rf_tuned = random_forest_hyperparameter_tuning(classification_results['random_forest'])
  lr_tuned = logistic_regression_hyperparameter_tuning(classification_results['logistic_regression'])
  hyperparameter_tuning_results = {
    'knn_tuned': knn_tuned,
    'random_forest_tuned': rf_tuned,
    'logistic_regression_tuned': lr_tuned,
  }

  print("\n*** HYPERPARAMETER TUNING SUMMARY ***")
  print(f"{'Classifier':<20} {'Before':<12} {'After':<12} {'Improvement':<15}")
  print("-"*60)
    
  print(f"{'k-NN':<20} {classification_results['knn']['accuracy']:<12.4f} {knn_tuned['tuned_accuracy']:<12.4f} {f'+{knn_tuned['improvement']:.4f} (+{knn_tuned['improvement']*100:.1f}%)':<15}")
  print(f"{'Random Forest':<20} {classification_results['random_forest']['accuracy']:<12.4f} {rf_tuned['tuned_accuracy']:<12.4f} {f'+{rf_tuned['improvement']:.4f} (+{rf_tuned['improvement']*100:.1f}%)':<15}")
  print(f"{'Logistic Regression':<20} {classification_results['logistic_regression']['accuracy']:<12.4f} {lr_tuned['tuned_accuracy']:<12.4f} {f'+{lr_tuned['improvement']:.4f} (+{lr_tuned['improvement']*100:.1f}%)':<15}")
    
  print("-"*60 + "\n")

  return hyperparameter_tuning_results

def main():
  # Load and split data
  data, train_data, test_data = load_and_split_data()

  # Run analysis using Hierarchical, Random Forest, and ...
  clustering_results = clustering_analysis(train_data)
  outlier_results = outlier_detection(train_data)
  feature_selection_results = feature_selection(train_data, test_data)
  classification_results = classification(train_data, test_data, feature_selection_results)
  tuning_results = hyperparameter_tuning(classification_results)

  return {
    'clustering': clustering_results,
    'outliers': outlier_results,
    'features': feature_selection_results,
    'classification': classification_results,
    'tuning': tuning_results
  }

if __name__ == "__main__":
  main()
