import numpy as np
import pandas as pd
from typing import List
from collections import Counter

from .decision_tree import DecisionTree  


class RandomForest:
    '''
    This is the code for the random forest classification, that builds up on the decision tree that 
    we implemented for the assignment 1. 
    '''
    def __init__(self, n_trees: int = 10, max_depth: int = None, min_samples_split: int = 2, max_features: int = None):
        '''
        n_trees: number of the trees in the random forest
        max_depth: maximum depth of each tree
        min_samples_split: minimum number of objects in each node 
        max_feature: number of maximum features we gonna use in each tree for splitting
        '''
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def bootstrap_sample(self, X: pd.DataFrame, y: pd.Series):
        
        # returns the sample with replacement for the bootstrapping 
        n_samples = len(X)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X.iloc[indices], y.iloc[indices]

    def fit(self, X: pd.DataFrame, y: pd.Series):

        self.trees = []
        # we will make up to n_trees number of trees
        # for each tree, we gonna train it on the bootstapped sample
        # and each tree gonna have random choice of the max_features number of attributes
        for _ in range(self.n_trees):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            
            if self.max_features:
                features = X_sample.columns
                selected_features = np.random.choice(features, size=self.max_features, replace=False)
                X_sample = X_sample[selected_features]

            tree.fit(X_sample, y_sample)
            self.trees.append((tree, X_sample.columns)) 

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        '''
        predict the label by taking the majority vote of the all trees in the random forest.
        '''
        tree_predictions = []
        for tree, features in self.trees:
            tree_predictions.append(tree.predict(X[features]))

        tree_predictions = np.array(tree_predictions).T
        majority_vote = [Counter(row).most_common(1)[0][0] for row in tree_predictions]
        return np.array(majority_vote)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> float:
        '''
        returns the accuracy score. 
        '''
        predictions = self.predict(X)
        return np.mean(predictions == y.to_numpy())
