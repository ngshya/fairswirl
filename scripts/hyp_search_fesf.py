import sys
sys.path.append('.')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import wandb
from numpy import abs, vstack, hstack, exp
from pickle import load
from argparse import ArgumentParser
from fairsslearn.metrics import get_metrics
from fairsslearn.models.fesf import FESF


# seed

my_seed = 1102


# arguments

parser = ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--split", type=str)
parser.add_argument("--sets", type=int, default=1)
parser.add_argument("--k", type=int, default=200)
parser.add_argument("--base_model", type=str, default="LR")
args = parser.parse_args()
print("Input parameters:", args)


# loading the dataset

path_file = "datasets/" + args.dataset + "/" + args.dataset + "_" + args.split + ".pickle"
assert os.path.exists(path_file), "Data file not found!"
with open(path_file, "rb") as f:
    dict_data = load(f)


# wandb

wb = wandb.init(config=args, reinit=True)


# preparing the training sets

X_train = dict_data["L0"]["X"]
y_train = dict_data["L0"]["y"]
s_train = dict_data["L0"]["s"]
y_train_one_hot = dict_data["L0"]["y_one_hot"]
s_train_one_hot = dict_data["L0"]["s_one_hot"]

for ss in range(args.sets-1):
    X_train = vstack((X_train, dict_data["L"+str(ss+1)]["X"]))
    y_train = hstack((y_train, dict_data["L"+str(ss+1)]["y"]))
    y_train_one_hot = vstack((y_train_one_hot, dict_data["L"+str(ss+1)]["y_one_hot"]))
    s_train = hstack((s_train, dict_data["L"+str(ss+1)]["s"]))
    s_train_one_hot = vstack((s_train_one_hot, dict_data["L"+str(ss+1)]["s_one_hot"]))

X_train_with_U = vstack((X_train, dict_data["U"]["X"]))
y_train_with_U = hstack((y_train, dict_data["U"]["y"]))
y_train_one_hot_with_U = vstack((y_train_one_hot, dict_data["U"]["y_one_hot"]))
s_train_with_U = hstack((s_train, dict_data["U"]["s"]))
s_train_one_hot_with_U = vstack((s_train_one_hot, dict_data["U"]["s_one_hot"]))


# FESF

model_fesf = FESF(K=args.k, base_model=args.base_model, seed=my_seed)

n_half_train = int(0.5*dict_data["U"]["X"].shape[0])

model_fesf.fit(
    Xl=X_train,
    yl=y_train,
    sl=s_train,
    Xu=dict_data["U"]["X"][:n_half_train,:],
    su=dict_data["U"]["s"][:n_half_train]
)

y_train_l_pred = model_fesf.predict(X_train)
y_train_u1_pred = model_fesf.predict(dict_data["U"]["X"][:n_half_train,:])
y_train_u2_pred = model_fesf.predict(dict_data["U"]["X"][n_half_train:,:])
y_train_pred = model_fesf.predict(X_train_with_U)
y_val_pred = model_fesf.predict(dict_data["V"]["X"])

acc_y_train_l, acc_s_train_l, mcc_y_train_l, mcc_s_train_l, sad_train_l, _, _ \
= get_metrics(
    y_pred=y_train_l_pred, 
    y_real=y_train, 
    s_pred=s_train,
    s_real=s_train,
)

acc_y_train_u1, acc_s_train_u1, mcc_y_train_u1, mcc_s_train_u1, sad_train_u1, _, _ \
= get_metrics(
    y_pred=y_train_u1_pred, 
    y_real=dict_data["U"]["y"][:n_half_train], 
    s_pred=dict_data["U"]["s"][:n_half_train],
    s_real=dict_data["U"]["s"][:n_half_train],
)

acc_y_train_u2, acc_s_train_u2, mcc_y_train_u2, mcc_s_train_u2, sad_train_u2, _, _ \
= get_metrics(
    y_pred=y_train_u2_pred, 
    y_real=dict_data["U"]["y"][n_half_train:], 
    s_pred=dict_data["U"]["s"][n_half_train:],
    s_real=dict_data["U"]["s"][n_half_train:],
)

acc_y_train, acc_s_train, mcc_y_train, mcc_s_train, sad_train, _, _ \
= get_metrics(
    y_pred=y_train_pred, 
    y_real=y_train_with_U, 
    s_pred=s_train_with_U,
    s_real=s_train_with_U,
)

acc_y_val, acc_s_val, mcc_y_val, mcc_s_val, sad_val, _, _ \
= get_metrics(
    y_pred=y_val_pred, 
    y_real=dict_data["V"]["y"], 
    s_pred=dict_data["V"]["s"],
    s_real=dict_data["V"]["s"],
)


# optimized metric

dis_mcc = mcc_y_val * exp(-30 * abs(sad_train_u2))


# wandb log

wb.log({
    "acc_y_train": acc_y_train, 
    "acc_y_train_l": acc_y_train_l, 
    "acc_y_train_u1": acc_y_train_u1, 
    "acc_y_train_u2": acc_y_train_u2, 
    "acc_y_val": acc_y_val, 
    "mcc_y_train": mcc_y_train, 
    "mcc_y_train_l": mcc_y_train_l, 
    "mcc_y_train_u1": mcc_y_train_u1, 
    "mcc_y_train_u2": mcc_y_train_u2, 
    "mcc_y_val": mcc_y_val, 
    "sad_train": sad_train, 
    "sad_train_l": sad_train_l, 
    "sad_train_u1": sad_train_u1, 
    "sad_train_u2": sad_train_u2, 
    "sad_val": sad_val, 
    "dis_mcc": dis_mcc,
})