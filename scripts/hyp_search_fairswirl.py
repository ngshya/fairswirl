import sys
sys.path.append('.')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import wandb
from sklearn.ensemble import RandomForestClassifier
from numpy import unique, abs, vstack, hstack, exp
from pickle import load
from tensorflow.compat.v1 import reset_default_graph
from tensorflow.keras.backend import clear_session
from argparse import ArgumentParser
from fairsslearn.models.fairswirl import FairSwiRL
from fairsslearn.metrics import get_metrics


# reset sessions and set the seed

reset_default_graph()
clear_session()
my_seed = 1102


# arguments

parser = ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--split", type=str)
parser.add_argument("--sets", type=int, default=1)
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
args = parser.parse_args()
print("Input parameters:", args)


# loading the dataset

path_file = "datasets/" + args.dataset + "/" + args.dataset + "_" + args.split + ".pickle"
assert os.path.exists(path_file), "Data file not found!"
with open(path_file, "rb") as f:
    dict_data = load(f)


# wandb

wb = wandb.init(config=args, reinit=True)


# FairSwiRL net architecture

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


# initializing FairSwiRL

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

# preparing the training set

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

# fit the FairSwiRL 

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

# get the predictions of FairSwiRL

y_train_pred = model_FairSwiRL.models["classifier"].predict(X_train_with_U)[:, 1]
s_train_pred = model_FairSwiRL.models["discriminator"].predict(X_train_with_U)[:, 1]

y_val_pred = model_FairSwiRL.models["classifier"].predict(dict_data["V"]["X"])[:, 1]
s_val_pred = model_FairSwiRL.models["discriminator"].predict(dict_data["V"]["X"])[:, 1]

# performance on training set

acc_y_train, acc_s_train, mcc_y_train, mcc_s_train, sad_train, _, _ \
= get_metrics(
    y_pred=(y_train_pred>0.5)+0, 
    y_real=y_train_with_U, 
    s_pred=(s_train_pred>0.5)+0,
    s_real=s_train_with_U,
)

# performance on validation set

acc_y_val, acc_s_val, mcc_y_val, mcc_s_val, sad_val, _, _ \
= get_metrics(
    y_pred=(y_val_pred>0.5)+0, 
    y_real=dict_data["V"]["y"], 
    s_pred=(s_val_pred>0.5)+0,
    s_real=dict_data["V"]["s"],
)


# training the Random Forest for S

n_half_train = int(0.5*X_train_with_U.shape[0])

model_rf = RandomForestClassifier()
model_rf.fit(
    X=model_FairSwiRL.models["embedder"].predict(X_train_with_U[:n_half_train,:]), 
    y=s_train_with_U[:n_half_train]
)
s_train_1_pred = model_rf.predict_proba(X=model_FairSwiRL.models["embedder"].predict(X_train_with_U[:n_half_train,:]))[:, 1]
s_train_2_pred = model_rf.predict_proba(X=model_FairSwiRL.models["embedder"].predict(X_train_with_U[n_half_train:,:]))[:, 1]
s_train_pred = model_rf.predict_proba(X=model_FairSwiRL.models["embedder"].predict(X_train_with_U))[:, 1]
s_val_pred = model_rf.predict_proba(X=model_FairSwiRL.models["embedder"].predict(dict_data["V"]["X"]))[:, 1]


# training the Random Forest for Y

model_rf = RandomForestClassifier()
model_rf.fit(
    X=model_FairSwiRL.models["embedder"].predict(X_train), 
    y=y_train
)
y_train_1_pred = model_rf.predict_proba(X=model_FairSwiRL.models["embedder"].predict(X_train_with_U[:n_half_train,:]))[:, 1]
y_train_2_pred = model_rf.predict_proba(X=model_FairSwiRL.models["embedder"].predict(X_train_with_U[n_half_train:,:]))[:, 1]
y_train_pred = model_rf.predict_proba(X=model_FairSwiRL.models["embedder"].predict(X_train_with_U))[:, 1]
y_val_pred = model_rf.predict_proba(X=model_FairSwiRL.models["embedder"].predict(dict_data["V"]["X"]))[:, 1]

# performance on training sets

acc_y_train_1_rf, acc_s_train_1_rf, mcc_y_train_1_rf, mcc_s_train_1_rf, sad_train_1_rf, _, _ \
= get_metrics(
    y_pred=(y_train_1_pred>0.5)+0, 
    y_real=y_train_with_U[:n_half_train], 
    s_pred=(s_train_1_pred>0.5)+0,
    s_real=s_train_with_U[:n_half_train],
)

acc_y_train_2_rf, acc_s_train_2_rf, mcc_y_train_2_rf, mcc_s_train_2_rf, sad_train_2_rf, _, _ \
= get_metrics(
    y_pred=(y_train_2_pred>0.5)+0, 
    y_real=y_train_with_U[n_half_train:], 
    s_pred=(s_train_2_pred>0.5)+0,
    s_real=s_train_with_U[n_half_train:],
)

acc_y_train_rf, acc_s_train_rf, mcc_y_train_rf, mcc_s_train_rf, sad_train_rf, _, _ \
= get_metrics(
    y_pred=(y_train_pred>0.5)+0, 
    y_real=y_train_with_U, 
    s_pred=(s_train_pred>0.5)+0,
    s_real=s_train_with_U,
)

# performance on validation set

acc_y_val_rf, acc_s_val_rf, mcc_y_val_rf, mcc_s_val_rf, sad_val_rf, _, _ \
= get_metrics(
    y_pred=(y_val_pred>0.5)+0, 
    y_real=dict_data["V"]["y"], 
    s_pred=(s_val_pred>0.5)+0,
    s_real=dict_data["V"]["s"],
)


# optimized metric

dis_mcc = mcc_y_val_rf * ( exp(-30 * abs(sad_train_2_rf)) + exp(-30 * abs(mcc_s_train_2_rf)) )


# wandb log

wb.log({
    "acc_s_train": acc_s_train,
    "acc_s_train_1_rf": acc_s_train_1_rf,
    "acc_s_train_2_rf": acc_s_train_2_rf,  
    "acc_s_train_rf": acc_s_train_rf, 
    "acc_s_val": acc_s_val,  
    "acc_s_val_rf": acc_s_val_rf, 
    "acc_y_train": acc_y_train, 
    "acc_y_train_rf": acc_y_train_rf, 
    "acc_y_train_1_rf": acc_y_train_1_rf, 
    "acc_y_train_2_rf": acc_y_train_2_rf, 
    "acc_y_val": acc_y_val, 
    "acc_y_val_rf": acc_y_val_rf,  
    "mcc_s_train": mcc_s_train,
    "mcc_s_train_rf": mcc_s_train_rf, 
    "mcc_s_train_1_rf": mcc_s_train_1_rf, 
    "mcc_s_train_2_rf": mcc_s_train_2_rf, 
    "mcc_s_val": mcc_s_val,  
    "mcc_s_val_rf": mcc_s_val_rf,  
    "mcc_y_train": mcc_y_train, 
    "mcc_y_train_rf": mcc_y_train_rf, 
    "mcc_y_train_1_rf": mcc_y_train_1_rf, 
    "mcc_y_train_2_rf": mcc_y_train_2_rf, 
    "mcc_y_val": mcc_y_val, 
    "mcc_y_val_rf": mcc_y_val_rf, 
    "sad_train": sad_train, 
    "sad_train_rf": sad_train_rf, 
    "sad_train_1_rf": sad_train_1_rf, 
    "sad_train_2_rf": sad_train_2_rf, 
    "sad_val": sad_val, 
    "sad_val_rf": sad_val_rf,  
    "dis_mcc": dis_mcc,
})