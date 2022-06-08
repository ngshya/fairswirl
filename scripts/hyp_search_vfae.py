import sys
sys.path.append('.')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import wandb
from sklearn.ensemble import RandomForestClassifier
from numpy import unique, abs, mean, vstack, hstack, exp, nan as np_nan, isnan
from pickle import load
from pandas import read_pickle
from tensorflow.compat.v1 import reset_default_graph
from tensorflow.keras.backend import clear_session
from argparse import ArgumentParser

from fairsslearn.models.vfae import VFAE
from fairsslearn.fairness_metrics import get_metrics
from fairsslearn.utils.mcc import best_ths_4_mcc


reset_default_graph()
clear_session()


my_seed = 1102


parser = ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--split", type=str)
parser.add_argument("--sets", type=int, default=1)

parser.add_argument("--z1", type=int)
parser.add_argument("--z2", type=int)
parser.add_argument("--enc1_hid", type=int)
parser.add_argument("--enc2_hid", type=int)
parser.add_argument("--dec1_hid", type=int)
parser.add_argument("--dec2_hid", type=int)
parser.add_argument("--us_hid", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--alpha", type=float)
parser.add_argument("--beta", type=float)
parser.add_argument("--D", type=int)
parser.add_argument("--gamma", type=float)

parser.add_argument("--epochs", type=int)

args = parser.parse_args()

print("Input parameters:", args)

# Loading tha dataset

path_file = "datasets/" + args.dataset + "/" + args.dataset + "_" + args.split + ".pickle"
assert os.path.exists(path_file), "Data file not found!"

wb = wandb.init(project=args.dataset+'-vfae-hyp-search', entity='mlgroup', config=args, reinit=True)

with open(path_file, "rb") as f:
    dict_data = load(f)

# Training VFAE

model = VFAE(
    z1=args.z1,
    z2=args.z2,
    enc1_hid=args.enc1_hid,
    enc2_hid=args.enc2_hid,
    dec1_hid=args.dec1_hid,
    dec2_hid=args.dec2_hid,
    us_hid=args.us_hid,
    lr=args.lr,
    alpha=args.alpha,
    beta=args.beta,
    D=args.D,
    gamma=args.gamma,
    seed=my_seed,
)

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
y_train_with_U_h = hstack((y_train, [np_nan] * dict_data["U"]["X"].shape[0]))
s_train_with_U = hstack((s_train, dict_data["U"]["s"]))
s_train_one_hot_with_U = vstack((s_train_one_hot, dict_data["U"]["s_one_hot"]))

model.fit(
    X_train=X_train_with_U, 
    y_train=y_train_with_U_h, 
    s_train=s_train_with_U, 
    epochs=args.epochs, 
    batch_size=64
)

ths_y = 0.5
ths_s = 0.5

emb = model.predict_emb(
    X=vstack((
        X_train_with_U, 
        dict_data["V"]["X"], 
        dict_data["T"]["X"]
    )), 
    s=hstack((
        s_train_with_U, 
        dict_data["V"]["s"],
        dict_data["T"]["s"]
    ))
)
emb_train = emb[:X_train_with_U.shape[0], :]
emb_val = emb[X_train_with_U.shape[0]:X_train_with_U.shape[0]+dict_data["V"]["X"].shape[0], :]
emb_test = emb[X_train_with_U.shape[0]+dict_data["V"]["X"].shape[0]:, :]

# Random Forest Training

n_half_train = int(0.5*X_train_with_U.shape[0])

model_rf = RandomForestClassifier()
model_rf.fit(
    X=emb_train[:n_half_train,:], 
    y=s_train_with_U[:n_half_train]
)
s_train_1_pred = model_rf.predict_proba(
    X=emb_train[:n_half_train,:]
)[:, 1]
s_train_2_pred = model_rf.predict_proba(
    X=emb_train[n_half_train:,:]
)[:, 1]
s_train_pred = model_rf.predict_proba(X=emb_train)[:, 1]
s_val_pred = model_rf.predict_proba(X=emb_val)[:, 1]
s_test_pred = model_rf.predict_proba(X=emb_test)[:, 1]

model_rf = RandomForestClassifier()
model_rf.fit(
    X=emb_train[~isnan(y_train_with_U_h)], 
    y=y_train
)

y_train_1_pred = model_rf.predict_proba(X=emb_train[:n_half_train,:])[:, 1]
y_train_2_pred = model_rf.predict_proba(X=emb_train[n_half_train:,:])[:, 1]
y_train_pred = model_rf.predict_proba(X=emb_train)[:, 1]
y_val_pred = model_rf.predict_proba(X=emb_val)[:, 1]
y_test_pred = model_rf.predict_proba(X=emb_test)[:, 1]

ths_y_rf = 0.5
ths_s_rf = 0.5

acc_y_train_1_rf, acc_s_train_1_rf, mcc_y_train_1_rf, mcc_s_train_1_rf, st_diff_train_1_rf, _, _ \
= get_metrics(
    y_pred=(y_train_1_pred>ths_y_rf)+0, 
    y_real=y_train_with_U[:n_half_train], 
    s_pred=(s_train_1_pred>ths_s_rf)+0,
    s_real=s_train_with_U[:n_half_train],
)

acc_y_train_2_rf, acc_s_train_2_rf, mcc_y_train_2_rf, mcc_s_train_2_rf, st_diff_train_2_rf, _, _ \
= get_metrics(
    y_pred=(y_train_2_pred>ths_y_rf)+0, 
    y_real=y_train_with_U[n_half_train:], 
    s_pred=(s_train_2_pred>ths_s_rf)+0,
    s_real=s_train_with_U[n_half_train:],
)

acc_y_train_rf, acc_s_train_rf, mcc_y_train_rf, mcc_s_train_rf, st_diff_train_rf, _, _ \
= get_metrics(
    y_pred=(y_train_pred>ths_y_rf)+0, 
    y_real=y_train_with_U, 
    s_pred=(s_train_pred>ths_s_rf)+0,
    s_real=s_train_with_U,
)

acc_y_val_rf, acc_s_val_rf, mcc_y_val_rf, mcc_s_val_rf, st_diff_val_rf, _, _ \
= get_metrics(
    y_pred=(y_val_pred>ths_y_rf)+0, 
    y_real=dict_data["V"]["y"], 
    s_pred=(s_val_pred>ths_s_rf)+0,
    s_real=dict_data["V"]["s"],
)

acc_y_test_rf, acc_s_test_rf, mcc_y_test_rf, mcc_s_test_rf, st_diff_test_rf, _, _ \
= get_metrics(
    y_pred=(y_test_pred>ths_y_rf)+0, 
    y_real=dict_data["T"]["y"], 
    s_pred=(s_test_pred>ths_s_rf)+0,
    s_real=dict_data["T"]["s"],
)

# Optimized metric

dis_mcc = mcc_y_val_rf * ( exp(-30 * abs(st_diff_train_2_rf)) + exp(-30 * abs(mcc_s_train_2_rf)) )

# wandb log

wb.log({
    "ths_y": ths_y,
    "ths_s": ths_s, 
    "ths_y_rf": ths_y_rf, 
    "ths_s_rf": ths_s_rf,
    "acc_s_train_1_rf": acc_s_train_1_rf,
    "acc_s_train_2_rf": acc_s_train_2_rf,  
    "acc_s_train_rf": acc_s_train_rf, 
    "acc_s_val_rf": acc_s_val_rf, 
    "acc_s_test_rf": acc_s_test_rf, 
    "acc_y_train_rf": acc_y_train_rf, 
    "acc_y_train_1_rf": acc_y_train_1_rf, 
    "acc_y_train_2_rf": acc_y_train_2_rf, 
    "acc_y_val_rf": acc_y_val_rf, 
    "acc_y_test_rf": acc_y_test_rf, 
    "mcc_s_train_rf": mcc_s_train_rf, 
    "mcc_s_train_1_rf": mcc_s_train_1_rf, 
    "mcc_s_train_2_rf": mcc_s_train_2_rf, 
    "mcc_s_val_rf": mcc_s_val_rf, 
    "mcc_s_test_rf": mcc_s_test_rf, 
    "mcc_y_train_rf": mcc_y_train_rf, 
    "mcc_y_train_1_rf": mcc_y_train_1_rf, 
    "mcc_y_train_2_rf": mcc_y_train_2_rf, 
    "mcc_y_val_rf": mcc_y_val_rf, 
    "mcc_y_test_rf": mcc_y_test_rf, 
    "st_diff_train_rf": st_diff_train_rf, 
    "st_diff_train_1_rf": st_diff_train_1_rf, 
    "st_diff_train_2_rf": st_diff_train_2_rf, 
    "st_diff_val_rf": st_diff_val_rf, 
    "st_diff_test_rf": st_diff_test_rf, 
    "dis_mcc": dis_mcc,
})
