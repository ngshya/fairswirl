from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelSpreading
from numpy.random import choice, random, seed as np_seed, permutation
from numpy import array, max, vstack, hstack, delete, unique


def fair_sample(x, n, cr=0.8, f=0.8):

    if n == 0:
        return array([])

    if x.shape[0] == 1:
        return array(list(x) * n)

    if x.shape[0] == 2:
        new_instances = array(list(x) * int(0.5*n+1))
        return new_instances[:n, :]
    
    dist_matrix = squareform(pdist(x))
    sample_idx = choice(x.shape[0], size=n, replace=True)

    new_instances = []

    for idx in sample_idx:
        knn_idx = dist_matrix[idx, :].argsort()[:3][1:]
        if random() < cr:
            new_instance = x[idx, :] + \
                f * (x[knn_idx[0], :] - x[knn_idx[1], :])
        else:
            new_instance = x[idx, :]
        new_instances.append(new_instance)

    new_instances = array(new_instances)

    return new_instances


def fair_smote(x, s, y, cr=0.8, f=0.8):
    
    # we suppose that x doesn't contain s and y. 

    x = array(x)
    s = array(s)
    y = array(y)
    
    bool_00 = array((s==0) & (y==0))
    bool_01 = array((s==0) & (y==1))
    bool_10 = array((s==1) & (y==0))
    bool_11 = array((s==1) & (y==1))

    n_00 = sum(bool_00)
    n_01 = sum(bool_01)
    n_10 = sum(bool_10)
    n_11 = sum(bool_11)

    if min((n_00, n_01, n_10, n_11)) > 0:

        max_n_instances = max((n_00, n_01, n_10, n_11))

        delta_n_00 = max_n_instances - n_00
        delta_n_01 = max_n_instances - n_01
        delta_n_10 = max_n_instances - n_10
        delta_n_11 = max_n_instances - n_11

        new_instances_00 = fair_sample(x=x[bool_00, :], n=delta_n_00, cr=cr, f=f)
        new_instances_01 = fair_sample(x=x[bool_01, :], n=delta_n_01, cr=cr, f=f)
        new_instances_10 = fair_sample(x=x[bool_10, :], n=delta_n_10, cr=cr, f=f)
        new_instances_11 = fair_sample(x=x[bool_11, :], n=delta_n_11, cr=cr, f=f)

        for new_instances in [
            new_instances_00, 
            new_instances_01, 
            new_instances_10, 
            new_instances_11
        ]:
            if (len(new_instances) > 0):
                x = vstack((x, new_instances))

        y = hstack((y, [0]*delta_n_00+[1]*delta_n_01+[0]*delta_n_10+[1]*delta_n_11))

    return x, y


def situation_testing(x, s, y, base_model="RF", seed=1102):

    # we suppose that x doesn't contain s and y. 

    if base_model == "LR":
        rf_s0 = LogisticRegression(random_state=seed)
        rf_s1 = LogisticRegression(random_state=seed)
    elif base_model == "RF":
        rf_s0 = RandomForestClassifier(random_state=seed)
        rf_s1 = RandomForestClassifier(random_state=seed)
    elif base_model == "SVC":
        rf_s0 = SVC(random_state=seed)
        rf_s1 = SVC(random_state=seed)
    elif base_model == "KNC":
        rf_s0 = KNeighborsClassifier()
        rf_s1 = KNeighborsClassifier()

    rf_s0.fit(x[s==0, :], y[s==0])
    rf_s1.fit(x[s==1, :], y[s==1])

    bool_fair = (rf_s0.predict(x) == rf_s1.predict(x)) & \
        (rf_s0.predict(x) == y)

    return x[bool_fair, :], s[bool_fair], y[bool_fair], \
        x[~bool_fair, :], s[~bool_fair], y[~bool_fair]


class FairSSL:

    def __init__(self, cr=0.8, f=0.8, base_model="RF", seed=1102):

        self.cr = cr
        self.f = f
        self.seed = seed
        self.base_model = base_model

        if base_model == "LR":
            self.final_model = LogisticRegression(random_state=self.seed)
        elif base_model == "RF":
            self.final_model = RandomForestClassifier(random_state=self.seed)
        elif base_model == "SVC":
            self.final_model = SVC(random_state=self.seed)
        elif base_model == "KNC":
            self.final_model = KNeighborsClassifier()


    def fit(self, xl, sl, yl, xu, su): 

        np_seed(self.seed)

        self.idx_del = []
        for j in range(xl.shape[1]):
            if (sum(xl[:, j] == sl) == xl.shape[0]) & \
            (sum(xu[:, j] == su) == xu.shape[0]):
                self.idx_del.append(j)
        xl = delete(xl, self.idx_del, axis=1)
        xu = delete(xu, self.idx_del, axis=1)

        # situation testing

        x_fair, s_fair, y_fair, x_unfair, s_unfair, y_unfair = situation_testing(xl, sl, yl, base_model=self.base_model, seed=self.seed)

        if len(unique(y_fair)) == 2:

            xu = vstack((xu, x_unfair))
            su = hstack((su, s_unfair))

        else:
            
            x_fair = xl
            s_fair = sl 
            y_fair = yl


        # Select the balanced fair dataset

        bool_00 = array((s_fair==0) & (y_fair==0))
        bool_01 = array((s_fair==0) & (y_fair==1))
        bool_10 = array((s_fair==1) & (y_fair==0))
        bool_11 = array((s_fair==1) & (y_fair==1))

        n_00 = sum(bool_00)
        n_01 = sum(bool_01)
        n_10 = sum(bool_10)
        n_11 = sum(bool_11)

        n_s = min((n_00, n_01, n_10, n_11))
        
        if n_s > 0:

            X00 = x_fair[bool_00, :]
            X01 = x_fair[bool_01, :]
            X10 = x_fair[bool_10, :]
            X11 = x_fair[bool_11, :]

            y00 = y_fair[bool_00]
            y01 = y_fair[bool_01]
            y10 = y_fair[bool_10]
            y11 = y_fair[bool_11]

            s00 = s_fair[bool_00]
            s01 = s_fair[bool_01]
            s10 = s_fair[bool_10]
            s11 = s_fair[bool_11]

            b00 = permutation([True] * n_s + [False] *   (n_00-n_s))
            b01 = permutation([True] * n_s + [False] *   (n_01-n_s))
            b10 = permutation([True] * n_s + [False] *   (n_10-n_s))
            b11 = permutation([True] * n_s + [False] *   (n_11-n_s))

            x = vstack(
                (
                    X00[b00, :], 
                    X01[b01, :], 
                    X10[b10, :], 
                    X11[b11, :], 
                )
            )
            y = hstack(
                (
                    y00[b00], 
                    y01[b01], 
                    y10[b10], 
                    y11[b11], 
                )
            )
            s = hstack(
                (
                    s00[b00], 
                    s01[b01], 
                    s10[b10], 
                    s11[b11], 
                )
            )

            xu = vstack(
                (
                    xu,
                    X00[~b00, :], 
                    X01[~b01, :], 
                    X10[~b10, :], 
                    X11[~b11, :], 
                )
            )
            su = hstack(
                (
                    su,
                    s00[~b00], 
                    s01[~b01], 
                    s10[~b10], 
                    s11[~b11], 
                )
            )

        else:

            x = x_fair 
            y = y_fair 
            s = s_fair

        # Label spreading

        model_ls = LabelSpreading()
        model_ls.fit(vstack((x, xu)), hstack((y, [-1]*xu.shape[0])))
        predu = model_ls.predict(xu)

        x = vstack((x, xu))
        y = hstack((y, predu))
        s = hstack((s, su))

        # fair smote

        x, y = fair_smote(x, s, y, self.cr, self.f)

        # final model

        self.final_model.fit(x, y)

    def predict(self, xt):

        xt = delete(xt, self.idx_del, axis=1)

        return self.final_model.predict(xt)