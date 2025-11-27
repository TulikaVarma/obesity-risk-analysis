import pandas as pd
import matplotlib.pyplot as plt
from classification.random_forest import RandomForest
from clustering.hierarchical_clustering import hierarchical_clustering
from outlier_detection.knn_outlier_detection import knn_outlier_detection
from feature_selection.mutual_information import mutual_information
from classification.knn_classification import knn_classification
from hyperparameter_tuning.knn_hyperparameter_tuning import knn_hyperparameter_tuning

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

def hierarchical_analysis(data, train_data, valid_data, test_data):
    # Hierarchical Analysis

    # 1. Clustering
    clustering = hierarchical_clustering(data)

    # 2. Outlier Detection
    knn_outlier = knn_outlier_detection(data, X_pca_2d=clustering['X_pca_2d'], k=5)

    # 3. Feature Selection
    feature_selection = mutual_information(data, train_data, valid_data, test_data)

    # 4. Classification
    knn_classification_results = knn_classification(train_data, valid_data, test_data, feature_selection)

    # 5. Hyperparameter Tuning
    tuning_results = knn_hyperparameter_tuning(knn_classification_results)

    return {
        'clustering': clustering,
        'outliers': knn_outlier,
        'features': feature_selection,
        'classification': knn_classification_results,
        'tuning': tuning_results
    }

def random_forest_predict(train_data, valid_data, test_data):
    # Random Forest Prediction
    X_train, y_train = train_data.drop(['NObeyesdad'], axis=1), train_data['NObeyesdad']
    X_valid, y_valid = valid_data.drop(['NObeyesdad'], axis=1), valid_data['NObeyesdad']
    X_test, y_test = test_data.drop(['NObeyesdad'], axis=1), test_data['NObeyesdad']

    # Initialize the classifiers
    rf = RandomForest(n_trees=3, max_depth=10, min_samples_split=10, max_features=5)
    #rf = RandomForest(n_trees=10, max_depth=5, min_samples_split=5, max_features=None)
    #nb = NaiveBayesClassifier(alpha=1)

    # Train Random Forest on the training data
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("=======Accuracy==============")
    print("Random Forest Training Accuracy: ", rf.evaluate(X_train, y_train))

    # Evaluate Random Forest on the test set
    print("Random Forest Test Accuracy: ", rf.evaluate(X_test, y_test))
    y_pred_rf = rf.predict(X_test)

def main():
    # Load and split data
    data, train_data, valid_data, test_data = load_and_split_data()

    # Run analysis using Hierarchical, Random Forest, and ...
    hierarchical_results = hierarchical_analysis(data, train_data, valid_data, test_data)
    rf_results = random_forest_predict(train_data, valid_data, test_data)

if __name__ == "__main__":
    main()