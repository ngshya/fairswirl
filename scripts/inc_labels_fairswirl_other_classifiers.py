import sys
sys.path.append('.')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from numpy import unique, mean, vstack, hstack
from pickle import load
from tensorflow.compat.v1 import reset_default_graph
from tensorflow.keras.backend import clear_session
from argparse import ArgumentParser
from fairsslearn.models.fairswirl import FairSwiRL
from fairsslearn.metrics import get_metrics


# reset session and seed

reset_default_graph()
clear_session()
my_seed = 1102


# arguments

parser = ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--split", type=str)
parser.add_argument("--ae_encoder_layers", type=int)
parser.add_argument("--cl_encoder_layers", type=int)
parser.add_argument("--di_encoder_layers", type=int)
parser.add_argument("--ae_encoder_1", type=int)
parser.add_argument("--ae_encoder_2", type=int)
parser.add_argument("--ae_encoder_3", type=int)
parser.add_argument("--ae_encoder_4", type=int)
parser.add_argument("--ae_encoder_5", type=int)
parser.add_argument("--cl_encoder_1", type=int)
parser.add_argument("--cl_encoder_2", type=int)
parser.add_argument("--cl_encoder_3", type=int)
parser.add_argument("--di_encoder_1", type=int)
parser.add_argument("--di_encoder_2", type=int)
parser.add_argument("--di_encoder_3", type=int)
parser.add_argument("--w_rec", type=float)
parser.add_argument("--w_cla", type=float)
parser.add_argument("--w_dis", type=float)
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--epochs", type=int)
parser.add_argument("--optimizer", type=str)
parser.add_argument("--sets", type=int)
parser.add_argument("--ths_y", type=float)
parser.add_argument("--ths_s", type=float)
parser.add_argument("--ths_y_rf", type=float)
parser.add_argument("--ths_s_rf", type=float)
parser.add_argument("--percentage", type=int, default=100)
args = parser.parse_args()
print("Input parameters:", args)


# loading the dataset

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
        y_train_one_hot = vstack((y_train_one_hot, dict_data["L"+str(ss+1)]["y_one_hot"]))
        s_train = hstack((s_train, dict_data["L"+str(ss+1)]["s"]))
        s_train_one_hot = vstack((s_train_one_hot, dict_data["L"+str(ss+1)]["s_one_hot"]))

X_train_with_U = vstack((X_train, dict_data["U"]["X"]))
y_train_with_U = hstack((y_train, dict_data["U"]["y"]))
y_train_one_hot_with_U = vstack((y_train_one_hot, dict_data["U"]["y_one_hot"]))
s_train_with_U = hstack((s_train, dict_data["U"]["s"]))
s_train_one_hot_with_U = vstack((s_train_one_hot, dict_data["U"]["s_one_hot"]))


# FairSwiRL network architecture

n_neurons_ae_encoder = [
    args.ae_encoder_1,
    args.ae_encoder_2,
    args.ae_encoder_3,
    args.ae_encoder_4,
    args.ae_encoder_5,
]

n_neurons_cl_encoder = [
    args.cl_encoder_1,
    args.cl_encoder_2,
    args.cl_encoder_3,
]

n_neurons_di_encoder = [
    args.di_encoder_1,
    args.di_encoder_2,
    args.di_encoder_3,
]

ae_encoder_size = sorted(n_neurons_ae_encoder, reverse=True)[:args.ae_encoder_layers]
ae_embedder_size = sorted(n_neurons_ae_encoder, reverse=True)[args.ae_encoder_layers]
ae_decoder_size = sorted(ae_encoder_size)
cl_encoder_size = sorted(n_neurons_cl_encoder, reverse=True)[:args.cl_encoder_layers]
di_encoder_size = sorted(n_neurons_di_encoder, reverse=True)[:args.di_encoder_layers]

wb.config["ae_encoder_size"] = ae_encoder_size
wb.config["ae_embedder_size"] = ae_embedder_size
wb.config["ae_decoder_size"] = ae_decoder_size
wb.config["cl_encoder_size"] = cl_encoder_size
wb.config["di_encoder_size"] = di_encoder_size


# FairSwiRL

model_FairSwiRL = FairSwiRL(
    input_size=dict_data["T"]["X"].shape[1],
    n_classes=len(unique(dict_data["T"]["y"])), 
    s_levels=len(unique(dict_data["T"]["s"])),
    ae_encoder_size=ae_encoder_size,
    ae_embedder_size=ae_embedder_size, 
    ae_decoder_size=ae_decoder_size,
    cl_encoder_size=cl_encoder_size, 
    di_encoder_size=di_encoder_size,
    w_rec=args.w_rec,
    w_cla=args.w_cla,
    w_dis=args.w_dis,
    optimizer=args.optimizer,
    learning_rate=args.learning_rate,
    seed=my_seed
)

model_FairSwiRL.fit(
    Xl=X_train, 
    yl=y_train_one_hot,
    sl=s_train_one_hot,
    Xu=dict_data["U"]["X"],
    su=dict_data["U"]["s_one_hot"],
    batch_size=64, 
    epochs=args.epochs, 
    Xt=dict_data["V"]["X"],
    yt=dict_data["V"]["y_one_hot"],
    st=dict_data["V"]["s_one_hot"],
)

y_train_pred = model_FairSwiRL.models["classifier"].predict(X_train_with_U)[:, 1]
s_train_pred = model_FairSwiRL.models["discriminator"].predict(X_train_with_U)[:, 1]

y_test_pred = model_FairSwiRL.models["classifier"].predict(dict_data["T"]["X"])[:, 1]
s_test_pred = model_FairSwiRL.models["discriminator"].predict(dict_data["T"]["X"])[:, 1]

acc_y_train, acc_s_train, mcc_y_train, mcc_s_train, f1_y_train, f1_s_train, sad_train, _, _ \
= get_metrics(
    y_pred=(y_train_pred>args.ths_y)+0, 
    y_real=y_train_with_U, 
    s_pred=(s_train_pred>args.ths_s)+0,
    s_real=s_train_with_U,
    compute_f1=True
)

acc_y_test, acc_s_test, mcc_y_test, mcc_s_test, f1_y_test, f1_s_test, sad_test, _, _ \
= get_metrics(
    y_pred=(y_test_pred>args.ths_y)+0, 
    y_real=dict_data["T"]["y"], 
    s_pred=(s_test_pred>args.ths_s)+0,
    s_real=dict_data["T"]["s"],
    compute_f1=True
)


# SVC Training on embeddings

model_svc = SVC()
model_svc.fit(
    X=model_FairSwiRL.models["embedder"].predict(X_train_with_U), 
    y=s_train_with_U
)
s_train_pred = model_svc.predict(X=model_FairSwiRL.models["embedder"].predict(X_train_with_U))
s_test_pred = model_svc.predict(X=model_FairSwiRL.models["embedder"].predict(dict_data["T"]["X"]))

model_svc = SVC()
model_svc.fit(
    X=model_FairSwiRL.models["embedder"].predict(X_train), 
    y=y_train
)

y_train_pred = model_svc.predict(X=model_FairSwiRL.models["embedder"].predict(X_train_with_U))
y_test_pred = model_svc.predict(X=model_FairSwiRL.models["embedder"].predict(dict_data["T"]["X"]))

acc_y_train_svc, acc_s_train_svc, mcc_y_train_svc, mcc_s_train_svc, f1_y_train_svc, f1_s_train_svc, sad_train_svc, _, _ \
= get_metrics(
    y_pred=y_train_pred, 
    y_real=y_train_with_U, 
    s_pred=s_train_pred,
    s_real=s_train_with_U,
    compute_f1=True
)

acc_y_test_svc, acc_s_test_svc, mcc_y_test_svc, mcc_s_test_svc, f1_y_test_svc, f1_s_test_svc, sad_test_svc, _, _ \
= get_metrics(
    y_pred=y_test_pred, 
    y_real=dict_data["T"]["y"], 
    s_pred=s_test_pred,
    s_real=dict_data["T"]["s"],
    compute_f1=True
)


# LogisticRegression Training on embeddings

model_lr = LogisticRegression()
model_lr.fit(
    X=model_FairSwiRL.models["embedder"].predict(X_train_with_U), 
    y=s_train_with_U
)
s_train_pred = model_lr.predict(X=model_FairSwiRL.models["embedder"].predict(X_train_with_U))
s_test_pred = model_lr.predict(X=model_FairSwiRL.models["embedder"].predict(dict_data["T"]["X"]))

model_lr = LogisticRegression()
model_lr.fit(
    X=model_FairSwiRL.models["embedder"].predict(X_train), 
    y=y_train
)

y_train_pred = model_lr.predict(X=model_FairSwiRL.models["embedder"].predict(X_train_with_U))
y_test_pred = model_lr.predict(X=model_FairSwiRL.models["embedder"].predict(dict_data["T"]["X"]))

acc_y_train_lr, acc_s_train_lr, mcc_y_train_lr, mcc_s_train_lr, f1_y_train_lr, f1_s_train_lr, sad_train_lr, _, _ \
= get_metrics(
    y_pred=y_train_pred, 
    y_real=y_train_with_U, 
    s_pred=s_train_pred,
    s_real=s_train_with_U,
    compute_f1=True
)

acc_y_test_lr, acc_s_test_lr, mcc_y_test_lr, mcc_s_test_lr, f1_y_test_lr, f1_s_test_lr, sad_test_lr, _, _ \
= get_metrics(
    y_pred=y_test_pred, 
    y_real=dict_data["T"]["y"], 
    s_pred=s_test_pred,
    s_real=dict_data["T"]["s"],
    compute_f1=True
)


# KNeighborsClassifier Training on embeddings

model_knc = KNeighborsClassifier()
model_knc.fit(
    X=model_FairSwiRL.models["embedder"].predict(X_train_with_U), 
    y=s_train_with_U
)
s_train_pred = model_knc.predict(X=model_FairSwiRL.models["embedder"].predict(X_train_with_U))
s_test_pred = model_knc.predict(X=model_FairSwiRL.models["embedder"].predict(dict_data["T"]["X"]))

model_knc = KNeighborsClassifier()
model_knc.fit(
    X=model_FairSwiRL.models["embedder"].predict(X_train), 
    y=y_train
)

y_train_pred = model_knc.predict(X=model_FairSwiRL.models["embedder"].predict(X_train_with_U))
y_test_pred = model_knc.predict(X=model_FairSwiRL.models["embedder"].predict(dict_data["T"]["X"]))

acc_y_train_knc, acc_s_train_knc, mcc_y_train_knc, mcc_s_train_knc, f1_y_train_knc, f1_s_train_knc, sad_train_knc, _, _ \
= get_metrics(
    y_pred=y_train_pred, 
    y_real=y_train_with_U, 
    s_pred=s_train_pred,
    s_real=s_train_with_U,
    compute_f1=True
)

acc_y_test_knc, acc_s_test_knc, mcc_y_test_knc, mcc_s_test_knc, f1_y_test_knc, f1_s_test_knc, sad_test_knc, _, _ \
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
    "mean_s_test": mean(dict_data["T"]["s"]), 
    "mean_y_test": mean(dict_data["T"]["y"]), 
    "acc_s_train": acc_s_train,
    "acc_s_test": acc_s_test, 
    "acc_y_train": acc_y_train, 
    "acc_y_test": acc_y_test, 
    "mcc_s_train": mcc_s_train,
    "mcc_s_test": mcc_s_test, 
    "mcc_y_train": mcc_y_train, 
    "mcc_y_test": mcc_y_test, 
    "f1_s_train": f1_s_train,
    "f1_s_test": f1_s_test, 
    "f1_y_train": f1_y_train, 
    "f1_y_test": f1_y_test, 
    "sad_train": sad_train, 
    "sad_test": sad_test, 
    "acc_s_train_svc": acc_s_train_svc,
    "acc_s_train_knc": acc_s_train_knc, 
    "acc_s_test_svc": acc_s_test_svc, 
    "acc_s_test_knc": acc_s_test_knc, 
    "acc_y_train_svc": acc_y_train_svc, 
    "acc_y_train_knc": acc_y_train_knc, 
    "acc_y_test_svc": acc_y_test_svc, 
    "acc_y_test_knc": acc_y_test_knc, 
    "mcc_s_train_svc": mcc_s_train_svc,
    "mcc_s_train_knc": mcc_s_train_knc, 
    "mcc_s_test_svc": mcc_s_test_svc, 
    "mcc_s_test_knc": mcc_s_test_knc, 
    "mcc_y_train_svc": mcc_y_train_svc, 
    "mcc_y_train_knc": mcc_y_train_knc, 
    "mcc_y_test_svc": mcc_y_test_svc, 
    "mcc_y_test_knc": mcc_y_test_knc,
    "f1_s_train_svc": f1_s_train_svc,
    "f1_s_train_knc": f1_s_train_knc, 
    "f1_s_test_svc": f1_s_test_svc, 
    "f1_s_test_knc": f1_s_test_knc, 
    "f1_y_train_svc": f1_y_train_svc, 
    "f1_y_train_knc": f1_y_train_knc, 
    "f1_y_test_svc": f1_y_test_svc, 
    "f1_y_test_knc": f1_y_test_knc,
    "sad_train_svc": sad_train_svc, 
    "sad_train_knc": sad_train_knc, 
    "sad_test_svc": sad_test_svc, 
    "sad_test_knc": sad_test_knc, 
    "acc_s_train_lr": acc_s_train_lr, 
    "acc_s_test_lr": acc_s_test_lr, 
    "acc_y_train_lr": acc_y_train_lr, 
    "acc_y_test_lr": acc_y_test_lr, 
    "mcc_s_train_lr": mcc_s_train_lr, 
    "mcc_s_test_lr": mcc_s_test_lr, 
    "mcc_y_train_lr": mcc_y_train_lr, 
    "mcc_y_test_lr": mcc_y_test_lr,
    "f1_s_train_lr": f1_s_train_lr, 
    "f1_s_test_lr": f1_s_test_lr, 
    "f1_y_train_lr": f1_y_train_lr, 
    "f1_y_test_lr": f1_y_test_lr,
    "sad_train_lr": sad_train_lr, 
    "sad_test_lr": sad_test_lr, 
})