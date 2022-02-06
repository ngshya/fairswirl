import sys
sys.path.append('.')
import os
import wandb
from sklearn.ensemble import RandomForestClassifier
from numpy import vstack, hstack
from pickle import load
from argparse import ArgumentParser
from fairsslearn.metrics import get_metrics


# seed

my_seed = 1102


# arguments

parser = ArgumentParser()
parser.add_argument("--sets", type=int)
parser.add_argument("--split", type=str)
parser.add_argument("--percentage", type=int, default=100)
args = parser.parse_args()
print("Input parameters:", args)


# Loading the dataset

path_file = "datasets/synthetic_h/synthetic_h_"+args.split+".pickle"
assert os.path.exists(path_file), "Data file not found!"
with open(path_file, "rb") as f:
    dict_data = load(f)


# wandb 

wb = wandb.init(config=args, reinit=True)


# preparing training sets

X_train = dict_data["L0"]["X"]
y_train = dict_data["L0"]["y"]
s_train = dict_data["L0"]["s"]

if args.percentage < 100:
    n_rows = int(args.percentage * X_train.shape[0] / 100.0)
    X_train = X_train[:n_rows, :]
    y_train = y_train[:n_rows]
    s_train = s_train[:n_rows]
else:
    for ss in range(args.sets-1):
        X_train = vstack((X_train, dict_data["L"+str(ss+1)]["X"]))
        y_train = hstack((y_train, dict_data["L"+str(ss+1)]["y"]))
        s_train = hstack((s_train, dict_data["L"+str(ss+1)]["s"]))

X_train_with_U = vstack((X_train, dict_data["U"]["X"]))
y_train_with_U = hstack((y_train, dict_data["U"]["y"]))
s_train_with_U = hstack((s_train, dict_data["U"]["s"]))


# random forest

model_rf = RandomForestClassifier()
model_rf.fit(
    X=X_train_with_U[:, :-2], 
    y=s_train_with_U
)
s_train_pred = model_rf.predict(X=X_train_with_U[:, :-2])
s_test_pred = model_rf.predict(X=dict_data["T"]["X"][:, :-2])

model_rf = RandomForestClassifier()
model_rf.fit(
    X=X_train[:, :-2], 
    y=y_train
)

y_train_pred = model_rf.predict(X=X_train_with_U[:, :-2])
y_test_pred = model_rf.predict(X=dict_data["T"]["X"][:, :-2])

acc_y_train_rf, acc_s_train_rf, mcc_y_train_rf, mcc_s_train_rf, f1_y_train_rf, f1_s_train_rf, st_diff_train_rf, _, _ \
= get_metrics(
    y_pred=y_train_pred, 
    y_real=y_train_with_U, 
    s_pred=s_train_pred,
    s_real=s_train_with_U,
    compute_f1=True
)

acc_y_test_rf, acc_s_test_rf, mcc_y_test_rf, mcc_s_test_rf, f1_y_test_rf, f1_s_test_rf, st_diff_test_rf, _, _ \
= get_metrics(
    y_pred=y_test_pred, 
    y_real=dict_data["T"]["y"], 
    s_pred=s_test_pred,
    s_real=dict_data["T"]["s"],
    compute_f1=True
)


# wandb log

wb.log({
    "labeled_instance": X_train.shape[0],
    "unlabeled_instances": dict_data["U"]["X"].shape[0],
    "test_instances": dict_data["T"]["X"].shape[0],
    "acc_s_train_rf": acc_s_train_rf,  
    "acc_s_test_rf": acc_s_test_rf, 
    "acc_y_train_rf": acc_y_train_rf,  
    "acc_y_test_rf": acc_y_test_rf, 
    "mcc_s_train_rf": mcc_s_train_rf, 
    "mcc_s_test_rf": mcc_s_test_rf, 
    "mcc_y_train_rf": mcc_y_train_rf, 
    "mcc_y_test_rf": mcc_y_test_rf,
    "st_diff_train_rf": st_diff_train_rf, 
    "st_diff_test_rf": st_diff_test_rf, 
    "f1_y_train_rf": f1_y_train_rf, 
    "f1_s_train_rf": f1_s_train_rf,
    "f1_y_test_rf": f1_y_test_rf, 
    "f1_s_test_rf": f1_s_test_rf,
})