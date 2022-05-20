import pandas as pd
from sklearn.model_selection import train_test_split

def export_results(prediction):
    result = pd.DataFrame(data={
        "Id": range(2000),
        "Category": prediction.astype(int)
    })

    result.to_csv('result.csv', index=False)

def load_dataset(filename, test_only=False, rs = 1):
    df = pd.read_csv("./dts/plain_dataset.csv")

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

    
    