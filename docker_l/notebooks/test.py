from scipy import stats
from sklearn.tree import DecisionTreeClassifier as tree_classifier
from random import seed
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# df = pd.read_feather("dts/np_dataset_train.ftr")
df = pd.read_csv("dts/np_dataset_train.csv")
df.info()


rs = 2

train, test = train_test_split(df, test_size=0.2, random_state=rs)


class random_forest():
    def __init__(self, number_of_trees=1, number_of_iteractions=1, max_depth=1, random_state=1):
        self.iteractions = number_of_iteractions
        self.number_of_trees = number_of_trees
        self.max_depth = max_depth
        self.forest = []
        self.random_state = random_state

    def bootstrap(self, x_samples):
        """ Create subsets of the input, and returns the indexes 
            from the original sample."""
        index = []

        sample_length = int(np.floor(len(x_samples)/self.number_of_trees))
        np.random.RandomState(self.random_state)

        index = np.random.randint(
            0, sample_length, (self.number_of_trees, sample_length))

        # if isinstance(x_samples, pd.DataFrame):
        #     x_samples = x_samples.to_numpy()
        
        # for i in range(self.number_of_trees):
        #     subsamples[i, :] = x_samples[subsamples[i, :], :]

        # for i in range(self.number_of_trees):
        #     subsamples.append(x_samples[np.random.randint(
        #         0, sample_length, (1, sample_length))])
            # index.append(np.random.randint(0, sample_length, [
            #              1, sample_length]))

        return index

    def predict(self, X):
        y = []
        for tree in self.forest:
            y.append(tree.predict(X))
        mode_ = stats.mode(y, axis=0)
        mode_ = np.transpose(mode_[0])
        return mode_

    def fit(self, x_samples, y_samples):
        idx = self.bootstrap(x_samples)

        for t in range(self.number_of_trees):
            tree = tree_classifier(
                criterion="gini", max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(x_samples.iloc[idx[t]], y_samples[idx[t]])
            self.forest.append(tree)

    # def train(self, x_samples, y_samples):
    #     self.bagging(x_samples, y_samples)


def flatten(t):
    return [item for item in t]


clf = random_forest(number_of_trees=2, number_of_iteractions=1, max_depth=1)


x = pd.DataFrame(train.iloc[:, :-1])
y = np.array(train.iloc[:, -1])

print(x.shape)
print(y.shape)

clf.fit(x, y)

print("FIT")
