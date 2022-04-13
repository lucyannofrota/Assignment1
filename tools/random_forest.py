import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as tree_classifier
from scipy import stats

class random_forest():
    def __init__(self, number_of_trees=1, number_of_iteractions=1, max_depth=1):
        self.iteractions= number_of_iteractions
        self.number_of_trees = number_of_trees
        self.max_depth = max_depth
        self.forest=[]

    def bootstrap(self, x_samples):
        """ Create subsets of the input, and returns the indexes 
            from the original sample."""
        index = []

        sample_length = len(x_samples)
        for i in range(self.number_of_trees):
            index.append(np.random.randint(0, sample_length, [1, sample_length]))
        return index

    def predict(self, X):
        y = []
        for tree in self.forest:
            y.append(tree.predict(X))
        mode_ = stats.mode(y, axis=0)
        mode_ = np.transpose(mode_[0])
        return mode_

    def bagging(self, x_samples, y_samples):
        idx = self.bootstrap(x_samples)

        for t in range(self.number_of_trees):
            tree = tree_classifier(criterion="gini", max_depth=self.max_depth)
            tree.fit(x_samples[flatten(idx[t])], y_samples[flatten(idx[t])])
            self.forest.append(tree)

    def train(self, x_samples, y_samples):
        self.bagging(x_samples, y_samples)
def flatten(t):
    return [item for item in t]