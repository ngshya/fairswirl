from pandas import read_excel
from numpy import array
from sklearn.preprocessing import MinMaxScaler


def get_card_data(path="datasets/raw/default of credit card clients.xls"):

    df = read_excel(path)
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df = df.loc[:, df.columns != "ID"]
    df["SEX"] = (df["SEX"].astype(int) == 1) + 0
    df["MARRIAGE"] = (df["MARRIAGE"].astype(int) == 1) + 0
    df["EDUCATION"] = df["EDUCATION"].astype(int).isin([1, 2, 3]) + 0
    df["default payment next month"] = (df["default payment next month"].astype(int) == 1) + 0

    X = array(df.loc[:, df.columns != "default payment next month"].astype(float))
    y = array(df["default payment next month"])
    s = array(df["EDUCATION"])

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y, s