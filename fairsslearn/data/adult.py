from pandas import read_csv, get_dummies, concat
from numpy import array
from sklearn.preprocessing import MinMaxScaler


def get_adult_data(path1, path2):

    df1 = read_csv(path1, header=None)
    df2 = read_csv(path2, header=None, skiprows=1)
    df = concat((df1, df2))
    df[9] = (df[9] == " Male") + 0
    df[14] = (df[14].isin([' >50K', ' >50K.'])) + 0
    df = df.rename(columns = {9: 'sex'})
    df = df.rename(columns = {14: 'target'})

    df = get_dummies(df)

    X = array(df.loc[:, df.columns != "target"])
    y = array(df["target"])
    s = array(df["sex"])

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y, s