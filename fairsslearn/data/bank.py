from pandas import read_csv, get_dummies
from numpy import array
from sklearn.preprocessing import MinMaxScaler


def get_bank_data(path="datasets/raw/bank-additional-full.csv"):

    df = read_csv(path, sep=";")
    df = get_dummies(df)

    del df["poutcome_nonexistent"]
    del df["poutcome_failure"]

    X = array(df.loc[:, (df.columns != "y_no") & (df.columns != "y_yes")])
    y = array(df["y_yes"])
    s = array(df["poutcome_success"])

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y, s