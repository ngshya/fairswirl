from numpy.random import \
    rand as np_rand, \
    choice as np_choice, \
    normal as np_normal, \
    seed as np_seed

from numpy import \
    vstack as np_vstack, \
    hstack as np_hstack, \
    transpose as np_transpose, \
    mean as np_mean
    

from sklearn.preprocessing import MinMaxScaler


def get_samples(n_samples, seed=1102, include_hidden=False):

    np_seed(seed)

    x0 = 2.0*(np_rand(n_samples)-0.5)
    x1 = 2.0*(np_rand(n_samples)-0.5)
    x2 = 2.0*(np_rand(n_samples)-0.5)
    x3 = 2.0*(np_rand(n_samples)-0.5)

    s = np_choice(a=[0,1], size=n_samples, replace=True, p=[0.5, 0.5])
    s1 = s + x0

    y = x1 + s1 + np_normal(0, 0.5, n_samples)
    y = (y > np_mean(y)) + 0

    if include_hidden:
        X1 = np_transpose(np_vstack((x0, x1, x2, x3, s, s1)))
    else:
        X1 = np_transpose(np_vstack((x1, x2, x3, s, s1)))
        
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X1)

    return X, y, s