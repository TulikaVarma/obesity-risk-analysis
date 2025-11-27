# Author: Arash Khoeini
# Email: akhoeini@sfu.ca
# Written for SFU CMPT 459

from typing import Optional, Sequence, Mapping
import numpy as np
import pandas as pd
from .node import Node
from typing import List, Tuple

class DecisionTree(object):
    def __init__(self, criterion: Optional['str'] = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = None):
        """
        :param criterion:
            The function to measure the quality of a split. Supported criteria are “gini” for the Gini
            impurity and “entropy” for the information gain.
        :param max_depth:
            The maximum depth of the tree.
        :param min_samples_split:
            The minimum number of samples required to be at a leaf node
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.criterion_func = self.entropy if criterion == 'entropy' else self.gini


    def fit(self, X: pd.DataFrame, y: pd.Series)->float:
        """
        :param X: data
        :param y: label column in X
        :return: accuracy of training dataset
        HINT1: You use self.tree to store the root of the tree
        HINT2: You should use self.split_node to split a node
        """
        # Your code

        single_class = False
        if y.nunique() == 1: 
            single_class = True

        node_class = None
        if y.nunique() == 1:
            node_class = str(y.unique()[0])

        self.tree = Node(len(X), node_class, 0, single_class)

        self.split_node(self.tree, X, y)


        return self.evaluate(X, y)

    def predict(self, X: pd.DataFrame)->np.ndarray:
        """
        :param X: data
        :return: predict the class for X.
        HINT1: You can use get_child_node method of Node class to traverse
        HINT2: You can use the mode of the class of the leaf node as the prediction
        HINT3: start traverasl from self.tree
        """
        predicts = []
        # Your code
        for index, row in X.iterrows():
            currentNode = self.tree
            while currentNode is not None and not currentNode.is_leaf:
                featureValue = row[currentNode.name]
                childNode = currentNode.get_child_node(featureValue)
                if childNode is None:
                    break
                currentNode = childNode
            predicts.append(currentNode.node_class)
        return np.array(predicts)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> int:
        """
        :param X: data
        :param y: labels
        :return: accuracy of predictions on X
        """
        preds = self.predict(X)
        acc = sum(preds == y) / len(preds)
        return acc


    def split_node(self, node: Node, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Splits the data in the node into child nodes based on the best feature.

        :param node: the current node to split
        :param X: data in the node
        :param y: labels in the node
        :return: None
        HINT1: Find the best feature to split the data in 'node'.
        HINT2: Use the criterion function (entropy/gini) to score the splits.
        HINT3: Split the data into child nodes
        HINT4: Recursively split the child nodes until the stopping condition is met (e.g., max_depth or single_class).
        """
        # your code
        featureList = [
            (column, pd.api.types.is_numeric_dtype(X[column]))
            for column in X.columns
        ]
        self.helper(node, X, y, featureList)

    def stopping_condition(self, node: Node) -> bool:
        """
        Checks if the stopping condition for splitting is met.

        :param node: The current node
        :return: True if stopping condition is met, False otherwise
        """
        # Check if the node is pure (all labels are the same)
        # Check if the maximum depth is reached

        if node.single_class:
            return True
        elif node.size < self.min_samples_split:
            return True
        elif node.depth >= self.max_depth:
            return True
        else:
            return False

    

    def gini(self, X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        """
        Returns gini index of the give feature

        :param X: data
        :param y: labels
        :param feature: name of the feature you want to use to get gini score
        :return:
        """

        numberOfClasses = y.value_counts()
        sampleCount = len(y)
        sumSquaredProbs = 0.0
        for count in numberOfClasses:
            prob = count / sampleCount
            sumSquaredProbs += prob ** 2
        giniIndex = 1 - sumSquaredProbs
        return giniIndex

    def entropy(self, X: pd.DataFrame, y: pd.Series, feature: str) ->float:
        """
        Returns entropy of the give feature

        :param X: data
        :param y: labels
        :param feature: name of the feature you want to use to get entropy score
        :return:
        """
        numberOfClasses = y.value_counts()
        sampleCount = len(y)
        entropy = 0.0
        for count in numberOfClasses:
            if count > 0:
                probability = count / sampleCount
                entropy -= probability * np.log2(probability)
        return entropy
    
    def splitScore(self, X: pd.DataFrame, y: pd.Series, feature: str, isNumerical: bool):
            totalSize = len(X)
            selectedThreshold = None
            impurity = 0
            
            if isNumerical:
                impurity = float('inf')
                uniqueValues = sorted(X[feature].unique())
                

                for i in range(1, len(uniqueValues)):
                    threshold = (uniqueValues[i-1] + uniqueValues[i]) / 2
                    xLess = X[X[feature] < threshold]
                    xGreater = X[X[feature] >= threshold]
                    yLess = y[X[feature] < threshold]
                    yGreater = y[X[feature] >= threshold]
                    if len(yGreater) == 0 or len(yLess) == 0:
                        continue  

                    greaterThanThresholdImpurity = self.criterion_func(xGreater, yGreater, None)
                    lessThanThresholdImpurity = self.criterion_func(xLess, yLess, None)
                    
                    weight_less = len(xLess) / totalSize
                    weight_greater = len(xGreater) / totalSize
                    weighted_impurity = (weight_less * lessThanThresholdImpurity) + (weight_greater * greaterThanThresholdImpurity)

                    if weighted_impurity < impurity:
                        impurity = weighted_impurity
                        selectedThreshold = threshold
            else:
                values = X[feature].unique()
                for value in values:
                    X_value = X[X[feature] == value]
                    weight = len(X_value) / totalSize
                    value_impurity = self.criterion_func(X_value, y[X[feature] == value], None)
                    impurity += weight * value_impurity

            if self.criterion_func == self.entropy:
                return self.criterion_func(X, y, None) - impurity , selectedThreshold
            else:
                return impurity, selectedThreshold


    def helper(self, node: Node, X: pd.DataFrame, y: pd.Series, features: List[Tuple[str, bool]]) -> None:
        
        if self.stopping_condition(node) == True:
            return
        
        if self.criterion_func == self.entropy:
            selectedScore = -float('inf')
        else:
            selectedScore = float('inf')

        
        selectedFeature = None
        isNumerical = None
        selectedThreshold = None

        for feature, isNum in features:
           
            scoreSplit, splitThreshold = self.splitScore(X,y,feature,isNum)

            if self.criterion_func == self.gini and scoreSplit < selectedScore:
                selectedScore = scoreSplit
                selectedThreshold = splitThreshold 
                selectedFeature = feature
                isNumerical = isNum
            elif self.criterion_func == self.entropy and scoreSplit > selectedScore:
                selectedScore = scoreSplit
                selectedThreshold = splitThreshold 
                selectedFeature = feature
                isNumerical = isNum

        if selectedFeature is None:
            return
        
        node.is_numerical = isNumerical
        node.name = selectedFeature
       
        if node.is_numerical:
            node.threshold = selectedThreshold
            
            yGreaterEqual = y[X[node.name] >= node.threshold]
            xGreaterEqual = X[X[node.name] >= node.threshold]
            
            XLess = X[X[node.name] < node.threshold]
            yLess = y[X[node.name] < node.threshold]

            isSingleClass = False
            if yGreaterEqual.nunique() == 1:
                isSingleClass = True
   
            greaterEqualChild = Node(xGreaterEqual.shape[0],yGreaterEqual.mode()[0],node.depth + 1,isSingleClass)
            
            isSingleClass = False
            if yLess.nunique() == 1:
                isSingleClass = True

            lessChild = Node(XLess.shape[0],yLess.mode()[0],node.depth + 1,isSingleClass)

            self.helper(lessChild,XLess,yLess,features)
            self.helper(greaterEqualChild,xGreaterEqual,yGreaterEqual,features)
        
            node.set_children({'ge': greaterEqualChild, 'l': lessChild}) 

        else:
            children={}
            updatedList = []
            for feature in features:
                if feature[0] != selectedFeature:
                    updatedList.append(feature)
            categoricalValues = X[selectedFeature].unique()

            for value in categoricalValues:
                yCategory = y[X[selectedFeature] == value]
                XCategory = X[X[selectedFeature] == value]
                
                isSingleClass = False
                if yCategory.nunique() == 1:
                    isSingleClass = True
                child = Node(XCategory.shape[0],yCategory.mode()[0],node.depth + 1,isSingleClass)
                
                self.helper(child,XCategory, yCategory, updatedList)
                children[value] = child
            if children:
                node.set_children(children) 


    
