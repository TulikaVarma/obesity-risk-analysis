import numpy as np
import pandas as pd

class NaiveBayesClassifier:
    def __init__(self, alpha=1):
        """
        Initialize the Naive Bayes Classifier with Laplacian smoothing parameter.
        alpha is the smoothing parameter (default=1)
        """
        self.alpha = alpha
        self.prior_probs = {}
        self.conditional_probs = {}

    def fit(self, X, y):
        """
        Train the Naive Bayes Classifier.
        """
        self.classes = np.unique(y)
        self.prior_probs = {cls: np.mean(y == cls) for cls in self.classes}

        self.conditional_probs = {}
        for cls in self.classes:
            class_data = X[y == cls]
            self.conditional_probs[cls] = {}
            for feature in X.columns:
                value_counts = class_data[feature].value_counts()
                # Number of distinct values for the feature
                mi = len(value_counts)  
                total = class_data.shape[0]
                # Calculating P(xi=aj|C=cls) with Laplacian smoothing
                self.conditional_probs[cls][feature] = {
                    value: (count + self.alpha) / (total + self.alpha * mi)
                    for value, count in value_counts.items()
                }
                #handling the unseen values
                self.conditional_probs[cls][feature]['unseen'] = self.alpha / (total + self.alpha * mi)

    def predict(self, X):
        """
        Predict the class for each sample in X.
        """
        predictions = []
        for _, row in X.iterrows():
            class_probs = {}
            for cls in self.classes:
                # we are going to claculate the  P(cls) * P(features|cls)
                class_prob = self.prior_probs[cls]
                for feature, value in row.items():
                    if value in self.conditional_probs[cls][feature]:
                        class_prob *= self.conditional_probs[cls][feature][value]
                    else:
                        # for the unseen values we gonna use the alpha / (all number of rows that had this class + alpha*number of distinctive value)
                        class_prob *= self.conditional_probs[cls][feature]['unseen']  
                class_probs[cls] = class_prob
            # the predict class is going to be the one with highest probablity 
            predictions.append(max(class_probs, key=class_probs.get))
        return predictions

    def evaluate(self, X, y):
        """
        Evaluate the model on the given dataset.
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy