import sys
sys.path.append('.')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import wandb
from sklearn.ensemble import RandomForestClassifier
from numpy import unique, abs, mean, vstack, hstack, isnan, nan as np_nan
from pickle import load
from tensorflow.compat.v1 import reset_default_graph
from tensorflow.keras.backend import clear_session
from argparse import ArgumentParser

from fairssae.models.dsa import DSA
from fairssae.models.fairssl import FairSSL
from fairssae.models.fesf import FESF
from fairssae.models.vfae import VFAE
from fairssae.fairness_metrics import get_metrics, statistical_difference


reset_default_graph()
clear_session()


def cm(s, y, m):
    dict_tmp = {}
    for ss in [0, 1]:
        for yy in [0, 1]:
            dict_tmp[m+"_s"+str(ss)+"y"+str(yy)] = sum((s==ss) & (y==yy))
    return dict_tmp


my_seed = 1102


parser = ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--split", type=str)
parser.add_argument("--labeled", type=int, default=100)
parser.add_argument("--extreme", type=int, default=0)
parser.add_argument("--wandb", type=int, default=0)
args = parser.parse_args()

print("Input parameters:", args)

# Loading the dataset

sig = "L100_L100x20_U10000_V100_T10000_seed"
if args.dataset == "compas":
    sig = "L100_L100x20_U1900_V100_T1900_seed"

path_file = "datasets/" + args.dataset + "/" + args.dataset + "_" + sig + args.split + ".pickle"
assert os.path.exists(path_file), "Data file not found!"

if args.wandb == 1:
    wb = wandb.init(project='inc-labels-experiments', entity='mlgroup', config=args, reinit=True)

with open(path_file, "rb") as f:
    dict_data = load(f)

dict_p = {}

X_train = dict_data["L0"]["X"]
y_train = dict_data["L0"]["y"]
s_train = dict_data["L0"]["s"]
y_train_one_hot = dict_data["L0"]["y_one_hot"]
s_train_one_hot = dict_data["L0"]["s_one_hot"]

if args.labeled < 100:
    n_rows = args.labeled
    X_train = X_train[:n_rows, :]
    y_train = y_train[:n_rows]
    s_train = s_train[:n_rows]
    y_train_one_hot = y_train_one_hot[:n_rows, :]
    s_train_one_hot = y_train_one_hot[:n_rows, :]
else:
    for ss in range(int(args.labeled/100)-1):
        X_train = vstack((X_train, dict_data["L"+str(ss+1)]["X"]))
        y_train = hstack((y_train, dict_data["L"+str(ss+1)]["y"]))
        y_train_one_hot = vstack((y_train_one_hot, dict_data["L"+str(ss+1)]["y_one_hot"]))
        s_train = hstack((s_train, dict_data["L"+str(ss+1)]["s"]))
        s_train_one_hot = vstack((s_train_one_hot, dict_data["L"+str(ss+1)]["s_one_hot"]))

bool_sy_equal_t = (y_train == s_train)
bool_sy_equal_x = (dict_data["X"]["y"] == dict_data["X"]["s"])

X_train = X_train[bool_sy_equal_t, :]
y_train = y_train[bool_sy_equal_t]
y_train_one_hot = y_train_one_hot[bool_sy_equal_t, :]
s_train = s_train[bool_sy_equal_t]
s_train_one_hot = s_train_one_hot[bool_sy_equal_t, :]

if args.extreme == 1:

    ntba = args.labeled - sum(bool_sy_equal_t)

    X_train = vstack((X_train, dict_data["X"]["X"][bool_sy_equal_x, :][:ntba, :]))
    y_train = hstack((y_train, dict_data["X"]["y"][bool_sy_equal_x][:ntba]))
    y_train_one_hot = vstack((y_train_one_hot, dict_data["X"]["y_one_hot"][bool_sy_equal_x, :][:ntba, :]))
    s_train = hstack((s_train, dict_data["X"]["s"][bool_sy_equal_x][:ntba]))
    s_train_one_hot = vstack((s_train_one_hot, dict_data["X"]["s_one_hot"][bool_sy_equal_x, :][:ntba, :]))

X_train_with_U = vstack((X_train, dict_data["U"]["X"]))
y_train_with_U = hstack((y_train, dict_data["U"]["y"]))
y_train_with_U_h = hstack((y_train, [np_nan] * dict_data["U"]["X"].shape[0]))
y_train_one_hot_with_U = vstack((y_train_one_hot, dict_data["U"]["y_one_hot"]))
s_train_with_U = hstack((s_train, dict_data["U"]["s"]))
s_train_one_hot_with_U = vstack((s_train_one_hot, dict_data["U"]["s_one_hot"]))

dict_p.update({
    "labeled_instances": X_train.shape[0],
    "unlabeled_instances": dict_data["U"]["X"].shape[0],
    "test_instances": dict_data["T"]["X"].shape[0],
    "mean_s_test": mean(dict_data["T"]["s"]), 
    "mean_y_test": mean(dict_data["T"]["y"]), 
    "st_diff_labeled": abs(statistical_difference(y_train, s_train)[0]),
})
dict_p.update(cm(dict_data["T"]["s"], dict_data["T"]["y"], "orig"))


# hyperparams

if args.dataset in ["synthetic", "synthetic_v2"]:
    n_neurons_ae_encoder = [13, 7, 9, 4, 6]   
    n_neurons_cl_encoder = [9, 10, 8]
    n_neurons_di_encoder = [8, 5, 2]
    ae_encoder_layers = 1
    cl_encoder_layers = 3
    di_encoder_layers = 2
    w_rec = 96.93324769529622
    w_cla = 4.899058647701775
    w_dis = 2.962127341021145
    ssae_epochs = 200
    optimizer = "Adam"
    learning_rate = 0.001
    fesf_k = 5
    fesf_base_model = "KNC"
    fairssl_cr = 0.2843853990982378
    fairssl_f = 0.9654557507853476
    fairssl_base_model = "SVC"
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
elif args.dataset == "adult":
    n_neurons_ae_encoder = [15, 12, 14, 9, 3]   
    n_neurons_cl_encoder = [10, 3, 10]
    n_neurons_di_encoder = [5, 16, 10]
    ae_encoder_layers = 3
    cl_encoder_layers = 2
    di_encoder_layers = 3
    w_rec = 67.5743604790751
    w_cla = 61.83146616402778
    w_dis = 38.8530807044244
    ssae_epochs = 200
    optimizer = "Adam"
    learning_rate = 0.001
    fesf_k = 9
    fesf_base_model = "RF"
    fairssl_cr = 0.4688682831971507
    fairssl_f = 0.95700173488701
    fairssl_base_model = "KNC"
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
elif args.dataset == "bank":
    n_neurons_ae_encoder = [8, 4, 12, 7, 8]   
    n_neurons_cl_encoder = [7, 2, 15]
    n_neurons_di_encoder = [13, 14, 10]
    ae_encoder_layers = 3
    cl_encoder_layers = 2
    di_encoder_layers = 1
    w_rec = 53.81067341989195
    w_cla = 58.53086074515461
    w_dis = 4.187386481910301
    ssae_epochs = 200
    optimizer = "Adam"
    learning_rate = 0.01
    fesf_k = 226
    fesf_base_model = "LR"
    fairssl_cr = 0.9256071743528388
    fairssl_f = 0.1965145613321823
    fairssl_base_model = "LR"
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
elif args.dataset == "card":
    n_neurons_ae_encoder = [12, 8, 5, 4, 8]   
    n_neurons_cl_encoder = [4, 13, 11]
    n_neurons_di_encoder = [4, 2, 15]
    ae_encoder_layers = 1
    cl_encoder_layers = 3
    di_encoder_layers = 1
    w_rec = 29.80800969788822
    w_cla = 77.37728042540292
    w_dis = 11.091398837427873
    ssae_epochs = 200
    optimizer = "Adam"
    learning_rate = 0.001
    fesf_k = 10
    fesf_base_model = "RF"
    fairssl_cr = 0.6376967575823875
    fairssl_f = 0.9026888168355712
    fairssl_base_model = "LR"
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
elif args.dataset == "compas":
    n_neurons_ae_encoder = [5, 11, 2, 5, 3]   
    n_neurons_cl_encoder = [16, 3, 13]
    n_neurons_di_encoder = [8, 12, 15]
    ae_encoder_layers = 4
    cl_encoder_layers = 2
    di_encoder_layers = 3
    w_rec = 73.17983896041078
    w_cla = 47.43188813135892
    w_dis = 41.99684659123851
    ssae_epochs = 200
    optimizer = "Adam"
    learning_rate = 0.01
    fesf_k = 181
    fesf_base_model = "SVC"
    fairssl_cr = 0.4016069161476691
    fairssl_f = 0.921773738991027
    fairssl_base_model = "RF"
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


# DSA

try:

    ths_y_rf = 0.5
    ths_s_rf = 0.5

    ae_encoder_size = sorted(n_neurons_ae_encoder, reverse=True)[:ae_encoder_layers]
    ae_embedder_size = sorted(n_neurons_ae_encoder, reverse=True)[ae_encoder_layers]
    ae_decoder_size = sorted(ae_encoder_size)
    cl_encoder_size = sorted(n_neurons_cl_encoder, reverse=True)[:cl_encoder_layers]
    di_encoder_size = sorted(n_neurons_di_encoder, reverse=True)[:di_encoder_layers]

    model_dsa = DSA(
        input_size=dict_data["T"]["X"].shape[1],
        n_classes=len(unique(dict_data["T"]["y"])), 
        s_levels=len(unique(dict_data["T"]["s"])),
        ae_encoder_size=ae_encoder_size,
        ae_embedder_size=ae_embedder_size, 
        ae_decoder_size=ae_decoder_size,
        cl_encoder_size=cl_encoder_size, 
        di_encoder_size=di_encoder_size,
        w_rec=w_rec,
        w_cla=w_cla,
        w_dis=w_dis,
        optimizer=optimizer,
        learning_rate=learning_rate,
        seed=my_seed
    )

    model_dsa.fit(
        Xl=X_train, 
        yl=y_train_one_hot,
        sl=s_train_one_hot,
        Xu=dict_data["U"]["X"],
        su=dict_data["U"]["s_one_hot"],
        batch_size=64, 
        epochs=ssae_epochs, 
        Xt=dict_data["V"]["X"],
        yt=dict_data["V"]["y_one_hot"],
        st=dict_data["V"]["s_one_hot"],
    )

    # Random Forest Training on embeddings

    model_rf = RandomForestClassifier()
    model_rf.fit(
        X=model_dsa.models["embedder"].predict(X_train_with_U), 
        y=s_train_with_U
    )
    s_train_pred = model_rf.predict_proba(X=model_dsa.models["embedder"].predict(X_train_with_U))[:, 1]
    s_test_pred = model_rf.predict_proba(X=model_dsa.models["embedder"].predict(dict_data["T"]["X"]))[:, 1]

    model_rf = RandomForestClassifier()
    model_rf.fit(
        X=model_dsa.models["embedder"].predict(X_train), 
        y=y_train
    )

    y_train_pred = model_rf.predict_proba(X=model_dsa.models["embedder"].predict(X_train_with_U))[:, 1]
    y_test_pred = model_rf.predict_proba(X=model_dsa.models["embedder"].predict(dict_data["T"]["X"]))[:, 1]

    acc_y_train_rf, acc_s_train_rf, mcc_y_train_rf, mcc_s_train_rf, f1_y_train_rf, f1_s_train_rf, st_diff_train_rf, _, _ \
    = get_metrics(
        y_pred=(y_train_pred>ths_y_rf)+0, 
        y_real=y_train_with_U, 
        s_pred=(s_train_pred>ths_s_rf)+0,
        s_real=s_train_with_U,
        compute_f1=True
    )

    acc_y_test_rf, acc_s_test_rf, mcc_y_test_rf, mcc_s_test_rf, f1_y_test_rf, f1_s_test_rf, st_diff_test_rf, _, _ \
    = get_metrics(
        y_pred=(y_test_pred>ths_y_rf)+0, 
        y_real=dict_data["T"]["y"], 
        s_pred=(s_test_pred>ths_s_rf)+0,
        s_real=dict_data["T"]["s"],
        compute_f1=True
    )

    dict_p.update({
        "acc_s_train_rf": acc_s_train_rf, 
        "acc_s_test_rf": acc_s_test_rf, 
        "acc_y_train_rf": acc_y_train_rf, 
        "acc_y_test_rf": acc_y_test_rf, 
        "mcc_s_train_rf": mcc_s_train_rf, 
        "mcc_s_test_rf": mcc_s_test_rf, 
        "mcc_y_train_rf": mcc_y_train_rf, 
        "mcc_y_test_rf": mcc_y_test_rf,
        "f1_s_train_rf": f1_s_train_rf, 
        "f1_s_test_rf": f1_s_test_rf, 
        "f1_y_train_rf": f1_y_train_rf, 
        "f1_y_test_rf": f1_y_test_rf,
        "st_diff_train_rf": st_diff_train_rf, 
        "st_diff_test_rf": st_diff_test_rf, 
    })
    dict_p.update(cm(dict_data["T"]["s"], (y_test_pred>ths_y_rf)+0, "rf"))

except Exception as e:
    print("Something went wrong:" + str(e))


# FESF

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

    acc_y_train_fesf, acc_s_train_fesf, mcc_y_train_fesf, mcc_s_train_fesf, f1_y_train_fesf, f1_s_train_fesf, st_diff_train_fesf, _, _ \
    = get_metrics(
        y_pred=y_train_pred, 
        y_real=y_train_with_U, 
        s_pred=s_train_with_U,
        s_real=s_train_with_U,
        compute_f1=True
    )

    acc_y_test_fesf, acc_s_test_fesf, mcc_y_test_fesf, mcc_s_test_fesf, f1_y_test_fesf, f1_s_test_fesf, st_diff_test_fesf, _, _ \
    = get_metrics(
        y_pred=y_test_pred, 
        y_real=dict_data["T"]["y"], 
        s_pred=dict_data["T"]["s"],
        s_real=dict_data["T"]["s"],
        compute_f1=True
    )

    dict_p.update({
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
        "st_diff_train_fesf": st_diff_train_fesf, 
        "st_diff_test_fesf": st_diff_test_fesf, 
    })
    dict_p.update(cm(dict_data["T"]["s"], y_test_pred, "fesf"))

except Exception as e:
    print("Something went wrong:" + str(e))


# FairSSL

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

    acc_y_train_fairssl, acc_s_train_fairssl, mcc_y_train_fairssl, mcc_s_train_fairssl, f1_y_train_fairssl, f1_s_train_fairssl, st_diff_train_fairssl, _, _ \
    = get_metrics(
        y_pred=y_train_pred, 
        y_real=y_train_with_U, 
        s_pred=s_train_with_U,
        s_real=s_train_with_U,
        compute_f1=True
    )

    acc_y_test_fairssl, acc_s_test_fairssl, mcc_y_test_fairssl, mcc_s_test_fairssl, f1_y_test_fairssl, f1_s_test_fairssl, st_diff_test_fairssl, _, _ \
    = get_metrics(
        y_pred=y_test_pred, 
        y_real=dict_data["T"]["y"], 
        s_pred=dict_data["T"]["s"],
        s_real=dict_data["T"]["s"],
        compute_f1=True
    )

    dict_p.update({
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
        "st_diff_train_fairssl": st_diff_train_fairssl, 
        "st_diff_test_fairssl": st_diff_test_fairssl, 
    })
    dict_p.update(cm(dict_data["T"]["s"], y_test_pred, "fairssl"))

except Exception as e:
    print("Something went wrong:" + str(e))


# VFAE

try:

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

    dict_p.update({
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
    dict_p.update(cm(dict_data["T"]["s"], (y_test_pred>ths_y)+0, "vfae"))

except Exception as e:
    print("Something went wrong:" + str(e))


if args.wandb == 1:
    wb.log(dict_p)
else:
    print(dict_p)
