import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from classification.random_forest import random_forest_classification
from clustering.hierarchical_clustering import hierarchical_clustering
from clustering.clarans import clarans_clustering
from outlier_detection.knn_outlier_detection import knn_outlier_detection
from outlier_detection.lof_outlier_detection import lof_outlier_detection
from feature_selection.mutual_information import mutual_information
from classification.knn_classification import knn_classification
from hyperparameter_tuning.knn_hyperparameter_tuning import knn_hyperparameter_tuning
from hyperparameter_tuning.random_forest_hyperparameter_tuning import random_forest_hyperparameter_tuning
from clustering.dbscan_clustering import dbscan_clustering
from outlier_detection.probabilistic_outlier_detection import probabilistic_outlier_detection
from feature_selection.lasso_regression import lasso_feature_selection
from classification.logistic_regression import logistic_regression_classification  

def load_and_split_data():
  # Load and split data consistently (60/20/20)
  data = pd.read_csv("cleaned_data.csv")
  data = data.sample(frac=1, random_state=30).reset_index(drop=True)
  
  train_size = int(0.6 * len(data))
  valid_size = int(0.2 * len(data))
  
  train_data = data.iloc[:train_size]
  valid_data = data.iloc[train_size:train_size + valid_size]
  test_data = data.iloc[train_size + valid_size:]
  
  print(f"Data Split:")
  print(f"Total Samples: {len(data)}")
  print(f"Training Samples: {len(train_data)} (60%)")
  print(f"Validation Samples: {len(valid_data)} (20%)")
  print(f"Test Samples: {len(test_data)} (20%)")
  
  return data, train_data, valid_data, test_data

def clustering_analysis(train_data):
  # Clustering Analysis
  print("1. CLUSTERING ANALYSIS")

  hierarchical_results = hierarchical_clustering(train_data)
  clarans_results = clarans_clustering(train_data)
  dbscan_results = dbscan_clustering(train_data)
  clustering_results = {
    'hierarchical': hierarchical_results,
    'clarans': clarans_results,
    'dbscan': dbscan_results,
  }

  # Compare all three clustering results and discussion
  # Compare using silhouette, calinski, davies
  # please give a bit of summarize from your own clustering result

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
  # Compare using number of outliers and percentage of outliers
  # please give a bit of summarize from your own outlier detection result

  return outlier_detection_results

def feature_selection(train_data, valid_data, test_data):
  # Feature Selection
  print("3. FEATURE SELECTION") 

  mi_results = mutual_information(train_data, valid_data, test_data)
  lasso_results = lasso_feature_selection(train_data, valid_data, test_data)
  feature_selection_results = {
    'mutual_information': mi_results,
    'lasso': lasso_results,
  }

  # Evaluate the model and for each model, compare with and without feature selection

  return feature_selection_results

def classification(train_data, valid_data, test_data, feature_selection_results):
  # Classification
  print("4. CLASSIFICATION")

  knn_results = knn_classification(train_data, valid_data, test_data, feature_selection_results['mutual_information'])
  lr_results = logistic_regression_classification(train_data, valid_data, test_data, feature_selection_results['mutual_information']) 
  rf_results = random_forest_classification(train_data, valid_data, test_data, feature_selection_results['mutual_information'])

  classification_results = {
    'knn': knn_results,
    'logistic_regression': lr_results,
    'random_forest': rf_results,
  }

  # Evaluate the models (classification results and discussion)
  
  return classification_results


def hyperparameter_tuning(classification_results):
  # Hyperparameter Tuning
  print("5. HYPERPARAMETER TUNING")

  knn_tuned = knn_hyperparameter_tuning(classification_results['knn'])
  rf_tuned = random_forest_hyperparameter_tuning(classification_results['random_forest'])
  # TODO: : add logistic regression hyperparameter tuning results

  hyperparameter_tuning_results = {
    'knn_tuned': knn_tuned,
    'random_forest_tuned': rf_tuned,
    # 'logistic_regression_tuned': logistic_regression_tuned,
  }
    
  # Evaluate the models (hyperparameter tuning results and discussion)

  return hyperparameter_tuning_results

def main():
  # Load and split data
  data, train_data, valid_data, test_data = load_and_split_data()

  # Run analysis using Hierarchical, Random Forest, and ...
  # hierarchical_results = hierarchical_analysis(data, train_data, valid_data, test_data)
  # rf_results = random_forest_predict(train_data, valid_data, test_data)
  clustering_results = clustering_analysis(train_data)
  outlier_results = outlier_detection(train_data)
  feature_selection_results = feature_selection(train_data, valid_data, test_data)
  classification_results = classification(train_data, valid_data, test_data, feature_selection_results)
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
