import sys
sys.path.append('.')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import wandb
from sklearn.ensemble import RandomForestClassifier
from numpy import vstack, hstack, exp, nan as np_nan, isnan, mean
from pickle import load
from tensorflow.compat.v1 import reset_default_graph
from tensorflow.keras.backend import clear_session
from argparse import ArgumentParser

from fairsslearn.models.vfae import VFAE
from fairsslearn.fairness_metrics import get_metrics


reset_default_graph()
clear_session()

my_seed = 1102


parser = ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--split", type=str)
parser.add_argument("--sets", type=int)
parser.add_argument("--percentage", type=int, default=100)
args = parser.parse_args()

if args.dataset == "synthetic": 
    z1 = 16
    z2 = 16
    enc1_hid = 4
    enc2_hid = 8
    dec1_hid = 8
    dec2_hid = 4
    us_hid = 32
    lr = 0.001
    alpha = 5.0
    beta = 5.0
    D = 100
    gamma = 1.0
    epochs = 50
if args.dataset == "adult": 
    z1 = 8
    z2 = 8
    enc1_hid = 8
    enc2_hid = 16
    dec1_hid = 32
    dec2_hid = 8
    us_hid = 16
    lr = 0.01
    alpha = 0.1
    beta = 10.0
    D = 5
    gamma = 1.0
    epochs = 50
if args.dataset == "bank": 
    z1 = 16
    z2 = 8
    enc1_hid = 4
    enc2_hid = 16
    dec1_hid = 8
    dec2_hid = 4
    us_hid = 8
    lr = 0.01
    alpha = 10.0
    beta = 5.0
    D = 100
    gamma = 1.0
    epochs = 50
if args.dataset == "compas": 
    z1 = 16
    z2 = 8
    enc1_hid = 4
    enc2_hid = 32
    dec1_hid = 8
    dec2_hid = 8
    us_hid = 32
    lr = 0.01
    alpha = 1.0
    beta = 10.0
    D = 200
    gamma = 1.0
    epochs = 50
if args.dataset == "card": 
    z1 = 8
    z2 = 16
    enc1_hid = 16
    enc2_hid = 4
    dec1_hid = 4
    dec2_hid = 8
    us_hid = 16
    lr = 0.001
    alpha = 0.5
    beta = 5.0
    D = 100
    gamma = 1.0
    epochs = 50

print("Input parameters:", args)

# Loading the dataset

path_file = "datasets/" + args.dataset + "/" + args.dataset + "_" + args.split + ".pickle"
assert os.path.exists(path_file), "Data file not found!"

wb = wandb.init(project='dsa-inc-labeled-vfae', entity='mlgroup', config=args, reinit=True)

with open(path_file, "rb") as f:
    dict_data = load(f)


X_train = dict_data["L0"]["X"]
y_train = dict_data["L0"]["y"]
s_train = dict_data["L0"]["s"]
y_train_one_hot = dict_data["L0"]["y_one_hot"]
s_train_one_hot = dict_data["L0"]["s_one_hot"]

if args.percentage < 100:
    n_rows = int(args.percentage * X_train.shape[0] / 100.0)
    X_train = X_train[:n_rows, :]
    y_train = y_train[:n_rows]
    s_train = s_train[:n_rows]
    y_train_one_hot = y_train_one_hot[:n_rows, :]
    s_train_one_hot = y_train_one_hot[:n_rows, :]
else:
    for ss in range(args.sets-1):
        X_train = vstack((X_train, dict_data["L"+str(ss+1)]["X"]))
        y_train = hstack((y_train, dict_data["L"+str(ss+1)]["y"]))
        y_train_one_hot = vstack((y_train_one_hot, dict_data["L"+str(ss+1)] ["y_one_hot"]))
        s_train = hstack((s_train, dict_data["L"+str(ss+1)]["s"]))
        s_train_one_hot = vstack((s_train_one_hot, dict_data["L"+str(ss+1)]["s_one_hot"]))

X_train_with_U = vstack((X_train, dict_data["U"]["X"]))
y_train_with_U = hstack((y_train, dict_data["U"]["y"]))
y_train_with_U_h = hstack((y_train, [np_nan] * dict_data["U"]["X"].shape[0]))
s_train_with_U = hstack((s_train, dict_data["U"]["s"]))
s_train_one_hot_with_U = vstack((s_train_one_hot, dict_data["U"]["s_one_hot"]))

# VFAE

model = VFAE(
    z1=z1,
    z2=z2,
    enc1_hid=enc1_hid,
    enc2_hid=enc2_hid,
    dec1_hid=dec1_hid,
    dec2_hid=dec2_hid,
    us_hid=us_hid,
    lr=lr,
    alpha=alpha,
    beta=beta,
    D=D,
    gamma=gamma,
    seed=my_seed,
)

ths_y = 0.5
ths_s = 0.5

acc_y_train_vfae = acc_s_train_vfae = mcc_y_train_vfae = mcc_s_train_vfae = f1_y_train_vfae = f1_s_train_vfae = st_diff_train_vfae = acc_y_test_vfae = acc_s_test_vfae = mcc_y_test_vfae = mcc_s_test_vfae = f1_y_test_vfae = f1_s_test_vfae = st_diff_test_vfae = None

try:

    model.fit(
        X_train=X_train_with_U, 
        y_train=y_train_with_U_h, 
        s_train=s_train_with_U, 
        epochs=epochs, 
        batch_size=64
    )

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

    model_rf = RandomForestClassifier()
    model_rf.fit(
        X=emb_train, 
        y=s_train_with_U
    )
    s_train_pred = model_rf.predict_proba(X=emb_train)[:, 1]
    s_val_pred = model_rf.predict_proba(X=emb_val)[:, 1]
    s_test_pred = model_rf.predict_proba(X=emb_test)[:, 1]

    model_rf = RandomForestClassifier()
    model_rf.fit(
        X=emb_train[~isnan(y_train_with_U_h)], 
        y=y_train
    )

    y_train_pred = model_rf.predict_proba(X=emb_train)[:, 1]
    y_val_pred = model_rf.predict_proba(X=emb_val)[:, 1]
    y_test_pred = model_rf.predict_proba(X=emb_test)[:, 1]

    acc_y_train_vfae, acc_s_train_vfae, mcc_y_train_vfae, mcc_s_train_vfae, f1_y_train_vfae, f1_s_train_vfae, st_diff_train_vfae, _, _ \
    = get_metrics(
        y_pred=(y_train_pred>ths_y)+0, 
        y_real=y_train_with_U, 
        s_pred=(s_train_pred>ths_s)+0,
        s_real=s_train_with_U,
        compute_f1=True
    )

    acc_y_test_vfae, acc_s_test_vfae, mcc_y_test_vfae, mcc_s_test_vfae, f1_y_test_vfae, f1_s_test_vfae, st_diff_test_vfae, _, _ \
    = get_metrics(
        y_pred=(y_test_pred>ths_y)+0, 
        y_real=dict_data["T"]["y"], 
        s_pred=(s_test_pred>ths_s)+0,
        s_real=dict_data["T"]["s"],
        compute_f1=True
    )
except Exception as e:
    print("Something went wrong:" + str(e))

# wandb log

wb.log({
    "labeled_instance": X_train.shape[0],
    "unlabeled_instances": dict_data["U"]["X"].shape[0],
    "test_instances": dict_data["T"]["X"].shape[0],
    "mean_s_test": mean(dict_data["T"]["s"]), 
    "mean_y_test": mean(dict_data["T"]["y"]), 
    "acc_s_train_vfae": acc_s_train_vfae, 
    "acc_s_test_vfae": acc_s_test_vfae, 
    "acc_y_train_vfae": acc_y_train_vfae, 
    "acc_y_test_vfae": acc_y_test_vfae, 
    "mcc_s_train_vfae": mcc_s_train_vfae, 
    "mcc_s_test_vfae": mcc_s_test_vfae, 
    "mcc_y_train_vfae": mcc_y_train_vfae, 
    "mcc_y_test_vfae": mcc_y_test_vfae,
    "f1_s_train_vfae": f1_s_train_vfae, 
    "f1_s_test_vfae": f1_s_test_vfae, 
    "f1_y_train_vfae": f1_y_train_vfae, 
    "f1_y_test_vfae": f1_y_test_vfae,
    "st_diff_train_vfae": st_diff_train_vfae, 
    "st_diff_test_vfae": st_diff_test_vfae, 
})
