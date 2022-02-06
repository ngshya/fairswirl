from numpy import array, abs, sum
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score


def ones_perc(l):
    l = array(l)
    if len(l) == 0:
        return 0.0
    else:
        return sum(l == 1) / len(l)


def zeros_perc(l):
    l = array(l)
    if len(l) == 0:
        return 0.0
    else:
        return sum(l == 0) / len(l)


def statistical_difference(y_pred, s):
    y_pred_s_0 = y_pred[s == 0]
    y_pred_s_1 = y_pred[s == 1]
    diff_0 = zeros_perc(y_pred_s_0) - zeros_perc(y_pred_s_1)
    diff_1 = ones_perc(y_pred_s_0) - ones_perc(y_pred_s_1)
    return [diff_0, diff_1]


def odds_difference(y_pred, y_real, s):
    y_pred_s_0_y_0 = y_pred[(s==0) & (y_real==0)]
    y_pred_s_0_y_1 = y_pred[(s==0) & (y_real==1)]
    y_pred_s_1_y_0 = y_pred[(s==1) & (y_real==0)]
    y_pred_s_1_y_1 = y_pred[(s==1) & (y_real==1)]
    diff_y_1 = zeros_perc(y_pred_s_0_y_1) - zeros_perc(y_pred_s_1_y_1)
    diff_y_0 = ones_perc(y_pred_s_0_y_0) - ones_perc(y_pred_s_1_y_0) 
    return [diff_y_0, diff_y_1]


def predictive_rate_difference(y_pred, y_real, s):
    y_real_s_0_y_pred_0 = y_real[(s==0) & (y_pred==0)]
    y_real_s_0_y_pred_1 = y_real[(s==0) & (y_pred==1)]
    y_real_s_1_y_pred_0 = y_real[(s==1) & (y_pred==0)]
    y_real_s_1_y_pred_1 = y_real[(s==1) & (y_pred==1)]
    diff_y_1 = ones_perc(y_real_s_0_y_pred_1) - ones_perc(y_real_s_1_y_pred_1)
    diff_y_0 = zeros_perc(y_real_s_0_y_pred_0) - zeros_perc(y_real_s_1_y_pred_0)
    return [diff_y_0, diff_y_1]


def get_fairness_metrics(
    y_pred, 
    y_real, 
    s, 
    verbose=True, 
    output=False
):
    st_diff = statistical_difference(y_pred=y_pred, s=s)
    od_diff = odds_difference(y_pred=y_pred, y_real=y_real, s=s)
    pr_diff = predictive_rate_difference(y_pred=y_pred, y_real=y_real, s=s)
    if verbose:
        print(
            "Stat.:", [round(x, 4) for x in st_diff], 
            "| Odds:", [round(x, 4) for x in od_diff], 
            "| P.R.:", [round(x, 4) for x in pr_diff]
        )
    if output:
        return [st_diff, od_diff, pr_diff]


def get_metrics(
    y_pred, 
    y_real, 
    s_pred,
    s_real, 
    compute_f1=False
): 

    y_pred = array(y_pred).astype(int)
    y_real = array(y_real).astype(int)
    s_pred = array(s_pred).astype(int)
    s_real = array(s_real).astype(int)

    st_diff, od_diff, pr_diff \
    = get_fairness_metrics(
        y_pred=y_pred, 
        y_real=y_real, 
        s=s_real,
        verbose=False, 
        output=True
    )

    acc_y = accuracy_score(y_true=y_real, y_pred=y_pred)
    acc_s = accuracy_score(y_true=s_real, y_pred=s_pred)

    mcc_y = matthews_corrcoef(y_true=y_real*2-1, y_pred=y_pred*2-1)
    mcc_s = matthews_corrcoef(y_true=s_real*2-1, y_pred=s_pred*2-1)

    if compute_f1:
        f1_y = f1_score(y_true=y_real, y_pred=y_pred)
        f1_s = f1_score(y_true=s_real, y_pred=s_pred)
        return acc_y, acc_s, mcc_y, mcc_s, f1_y, f1_s, abs(st_diff[0]), sum(abs(od_diff)), sum(abs(pr_diff))

    return acc_y, acc_s, mcc_y, mcc_s, abs(st_diff[0]), sum(abs(od_diff)), sum(abs(pr_diff))