import pandas as pd
import matplotlib.pyplot as plt
from classification.random_forest import RandomForest

def predict():
    # Load and preprocess data
    data = pd.read_csv("cleaned_data.csv")

    # Shuffle the dataset
    data = data.sample(frac=1, random_state=30).reset_index(drop=True)

    # Split the dataset (60/20/20)
    train_size = int(0.6 * len(data))
    valid_size = int(0.2 * len(data))

    #train_data = combined_data.iloc[:train_size] for the hyper parameter running uncomment this line and comment next line
    train_data = data.iloc[:valid_size]
    valid_data = data.iloc[train_size:train_size + valid_size]
    test_data = data.iloc[train_size + valid_size:]

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


if __name__ == "__main__":
    predict()