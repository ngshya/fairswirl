import sys
sys.path.append('.')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import wandb
from numpy import mean, vstack, hstack
from pickle import load
from argparse import ArgumentParser
from fairsslearn.models.fairssl import FairSSL
from fairsslearn.models.fesf import FESF
from fairsslearn.metrics import get_metrics


# seed

my_seed = 1102


# arguments 

parser = ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--split", type=str)
parser.add_argument("--sets", type=int)
parser.add_argument("--percentage", type=int, default=100)
args = parser.parse_args()


# hyperparameters

if args.dataset == "synthetic": 
    fesf_k = 5
    fesf_base_model = "KNC"
    fairssl_cr = 0.2843853990982378
    fairssl_f = 0.9654557507853476
    fairssl_base_model = "SVC"
if args.dataset == "adult": 
    fesf_k = 9
    fesf_base_model = "RF"
    fairssl_cr = 0.4688682831971507
    fairssl_f = 0.95700173488701
    fairssl_base_model = "KNC"
if args.dataset == "bank": 
    fesf_k = 226
    fesf_base_model = "LR"
    fairssl_cr = 0.9256071743528388
    fairssl_f = 0.1965145613321823
    fairssl_base_model = "LR"
if args.dataset == "compas": 
    fesf_k = 181
    fesf_base_model = "SVC"
    fairssl_cr = 0.4016069161476691
    fairssl_f = 0.921773738991027
    fairssl_base_model = "RF"
if args.dataset == "card": 
    fesf_k = 10
    fesf_base_model = "RF"
    fairssl_cr = 0.6376967575823875
    fairssl_f = 0.9026888168355712
    fairssl_base_model = "LR"

print("Input parameters:", args)


# Loading the dataset

path_file = "datasets/" + args.dataset + "/" + args.dataset + "_" + args.split + ".pickle"
assert os.path.exists(path_file), "Data file not found!"
with open(path_file, "rb") as f:
    dict_data = load(f)


# wandb

wb = wandb.init(config=args, reinit=True)


# preparing training sets

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
y_train_one_hot_with_U = vstack((y_train_one_hot, dict_data["U"]["y_one_hot"]))
s_train_with_U = hstack((s_train, dict_data["U"]["s"]))
s_train_one_hot_with_U = vstack((s_train_one_hot, dict_data["U"]["s_one_hot"]))


# FESF

acc_y_train_fesf = acc_s_train_fesf = mcc_y_train_fesf = mcc_s_train_fesf = f1_y_train_fesf = f1_s_train_fesf = sad_train_fesf = acc_y_test_fesf = acc_s_test_fesf = mcc_y_test_fesf = mcc_s_test_fesf = f1_y_test_fesf = f1_s_test_fesf = sad_test_fesf = None

try:
    model_fesf = FESF(K=fesf_k, base_model=fesf_base_model)
    model_fesf.fit(
        Xl=X_train,
        yl=y_train,
        sl=s_train,
        Xu=dict_data["U"]["X"],
        su=dict_data["U"]["s"]
    )
    y_train_pred = model_fesf.predict(X_train_with_U)
    y_test_pred = model_fesf.predict(dict_data["T"]["X"])

    acc_y_train_fesf, acc_s_train_fesf, mcc_y_train_fesf, mcc_s_train_fesf, f1_y_train_fesf, f1_s_train_fesf, sad_train_fesf, _, _ \
    = get_metrics(
        y_pred=y_train_pred, 
        y_real=y_train_with_U, 
        s_pred=s_train_with_U,
        s_real=s_train_with_U,
        compute_f1=True
    )

    acc_y_test_fesf, acc_s_test_fesf, mcc_y_test_fesf, mcc_s_test_fesf, f1_y_test_fesf, f1_s_test_fesf, sad_test_fesf, _, _ \
    = get_metrics(
        y_pred=y_test_pred, 
        y_real=dict_data["T"]["y"], 
        s_pred=dict_data["T"]["s"],
        s_real=dict_data["T"]["s"],
        compute_f1=True
    )
except Exception as e:
    print("Something went wrong:" + str(e))


# FairSSL

acc_y_train_fairssl = acc_s_train_fairssl = mcc_y_train_fairssl = mcc_s_train_fairssl = f1_y_train_fairssl = f1_s_train_fairssl = sad_train_fairssl = acc_y_test_fairssl = acc_s_test_fairssl = mcc_y_test_fairssl = mcc_s_test_fairssl = f1_y_test_fairssl = f1_s_test_fairssl = sad_test_fairssl = None

try:
    model_fairssl = FairSSL(cr=fairssl_cr, f=fairssl_f, base_model=fairssl_base_model)
    model_fairssl.fit(
        xl=X_train,
        yl=y_train,
        sl=s_train,
        xu=dict_data["U"]["X"],
        su=dict_data["U"]["s"]
    )
    y_train_pred = model_fairssl.predict(X_train_with_U)
    y_test_pred = model_fairssl.predict(dict_data["T"]["X"])

    acc_y_train_fairssl, acc_s_train_fairssl, mcc_y_train_fairssl, mcc_s_train_fairssl, f1_y_train_fairssl, f1_s_train_fairssl, sad_train_fairssl, _, _ \
    = get_metrics(
        y_pred=y_train_pred, 
        y_real=y_train_with_U, 
        s_pred=s_train_with_U,
        s_real=s_train_with_U,
        compute_f1=True
    )

    acc_y_test_fairssl, acc_s_test_fairssl, mcc_y_test_fairssl, mcc_s_test_fairssl, f1_y_test_fairssl, f1_s_test_fairssl, sad_test_fairssl, _, _ \
    = get_metrics(
        y_pred=y_test_pred, 
        y_real=dict_data["T"]["y"], 
        s_pred=dict_data["T"]["s"],
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
    "acc_s_train_fesf": acc_s_train_fesf, 
    "acc_s_test_fesf": acc_s_test_fesf, 
    "acc_y_train_fesf": acc_y_train_fesf, 
    "acc_y_test_fesf": acc_y_test_fesf, 
    "mcc_s_train_fesf": mcc_s_train_fesf, 
    "mcc_s_test_fesf": mcc_s_test_fesf, 
    "mcc_y_train_fesf": mcc_y_train_fesf, 
    "mcc_y_test_fesf": mcc_y_test_fesf,
    "f1_s_train_fesf": f1_s_train_fesf, 
    "f1_s_test_fesf": f1_s_test_fesf, 
    "f1_y_train_fesf": f1_y_train_fesf, 
    "f1_y_test_fesf": f1_y_test_fesf,
    "sad_train_fesf": sad_train_fesf, 
    "sad_test_fesf": sad_test_fesf, 
    "acc_s_train_fairssl": acc_s_train_fairssl,
    "acc_s_test_fairssl": acc_s_test_fairssl, 
    "acc_y_train_fairssl": acc_y_train_fairssl, 
    "acc_y_test_fairssl": acc_y_test_fairssl, 
    "mcc_s_train_fairssl": mcc_s_train_fairssl,
    "mcc_s_test_fairssl": mcc_s_test_fairssl, 
    "mcc_y_train_fairssl": mcc_y_train_fairssl, 
    "mcc_y_test_fairssl": mcc_y_test_fairssl, 
    "f1_s_train_fairssl": f1_s_train_fairssl,
    "f1_s_test_fairssl": f1_s_test_fairssl, 
    "f1_y_train_fairssl": f1_y_train_fairssl, 
    "f1_y_test_fairssl": f1_y_test_fairssl, 
    "sad_train_fairssl": sad_train_fairssl, 
    "sad_test_fairssl": sad_test_fairssl, 
})
