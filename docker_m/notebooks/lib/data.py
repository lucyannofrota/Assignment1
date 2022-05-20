import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import Dataset
import torch


def export_results(prediction):
    print(prediction)
    result = pd.DataFrame(data={
        "Id": range(prediction.shape[0]),
        "Category": prediction.astype(input())
    }, index=None)

    result.to_csv('result.csv', index=False)

def load_dataset(filename, test_only=False, rs = 1):
    df = pd.read_csv(filename)

    if not test_only:
        train, test = train_test_split(df, test_size=0.2, random_state=rs)
        
        x_test  = pd.DataFrame(test.iloc[:, :-1])
        y_test  = np.array(test.iloc[:, -1])

        x_train = pd.DataFrame(train.iloc[:, :-1])
        y_train = np.array(train.iloc[:, -1])
        return x_train, y_train, x_test, y_test
    else:
        x_test = df.iloc[:, :-1]
    return x_test


class dataset(Dataset):
    def __init__(self, x, y, sc = None):
        if sc is None:
            sc = StandardScaler()
            sc.fit(x)
        self.sc = sc
        x = self.sc.transform(x)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.x = torch.tensor(x, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
        return self.x[index], self.y[index]