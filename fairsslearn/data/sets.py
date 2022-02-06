from numpy.random import seed as np_seed, shuffle
from numpy import array, unique, eye


def luvt(
    X, 
    y, 
    s, 
    n_labeled_0,
    n_labeled_delta, 
    n_labeled_sets,
    n_unlabeled,
    n_val,  
    n_test, 
    seed=1102
):

    np_seed(seed)

    n_tot = X.shape[0]
    n_unused = n_tot - n_labeled_0 - n_labeled_delta * n_labeled_sets - n_unlabeled - n_val - n_test

    assert n_unused >= 0, "Not sufficient number of instances"

    n_classes = len(unique(y))
    n_levels = len(unique(s))
    y_one_hot = eye(n_classes)[y,:]
    s_one_hot = eye(n_levels)[s, :]

    l_labeled_sets = ["L"+str(i+1) for i in range(n_labeled_sets)]

    l = array(
        ["L0"]*n_labeled_0 + \
        l_labeled_sets * n_labeled_delta + \
        ["U"]*n_unlabeled + \
        ["V"]*n_val + \
        ["T"]*n_test + \
        ["X"]*n_unused
    )
    shuffle(l)

    dict_sets = {}
    for t in unique(l):
        l_bool = l == t
        dict_sets[t] = {}
        dict_sets[t]["X"] = X[l_bool, :]
        dict_sets[t]["y"] = y[l_bool]
        dict_sets[t]["y_one_hot"] = y_one_hot[l_bool, :]
        dict_sets[t]["s"] = s[l_bool]
        dict_sets[t]["s_one_hot"] = s_one_hot[l_bool, :]

    return dict_sets