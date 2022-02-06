import sys
sys.path.append('.')

from argparse import ArgumentParser
from pathlib import Path
from pickle import dump

from fairsslearn.data.synthetic import get_samples
from fairsslearn.data.adult import get_adult_data
from fairsslearn.data.bank import get_bank_data
from fairsslearn.data.card import get_card_data
from fairsslearn.data.compas import get_compas_data
from fairsslearn.data.sets import luvt


parser = ArgumentParser()
parser.add_argument(
    "--dataset", 
    choices=["synthetic", "synthetic_h", "adult", "bank", "card", "compas"], 
    type=str, 
    default="synthetic"
)
parser.add_argument("--n_labeled_0", type=int, default=100)
parser.add_argument("--n_labeled_delta", type=int, default=100)
parser.add_argument("--n_labeled_sets", type=int, default=20)
parser.add_argument("--n_unlabeled", type=int, default=10000)
parser.add_argument("--n_val", type=int, default=100)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--seed", type=int, default=1102)
args = parser.parse_args()



n_tot = args.n_labeled_0+\
    args.n_labeled_delta*args.n_labeled_sets+\
    args.n_unlabeled+\
    args.n_val+\
    args.n_test



if args.dataset == "synthetic":
    X, y, s = get_samples(
        n_samples=n_tot, 
        seed=args.seed, 
        include_hidden=False
    )
elif args.dataset == "synthetic_h":
    X, y, s = get_samples(
        n_samples=n_tot, 
        seed=args.seed, 
        include_hidden=True
    )
elif args.dataset == "adult":
    X, y, s = get_adult_data(
        path1="datasets/raw/adult.data", 
        path2="datasets/raw/adult.test"
    )
elif args.dataset == "bank":
    X, y, s = get_bank_data(path="datasets/raw/bank-additional-full.csv")
elif args.dataset == "card":
    X, y, s = get_card_data(path="datasets/raw/default of credit card clients.xls")
elif args.dataset == "compas":
    X, y, s = get_compas_data(path="datasets/raw/compas-scores-two-years.csv")



assert X.shape[0] >= n_tot, "Not sufficient instances."



dict_data = luvt(
    X=X, 
    y=y, 
    s=s, 
    n_labeled_0=args.n_labeled_0,
    n_labeled_delta=args.n_labeled_delta, 
    n_labeled_sets=args.n_labeled_sets,
    n_unlabeled=args.n_unlabeled,
    n_val=args.n_val,  
    n_test=args.n_test, 
    seed=args.seed
)



path_folder = "datasets/" + args.dataset + "/" 
Path(path_folder).mkdir(parents=True, exist_ok=True)
path_file = path_folder + \
    args.dataset + "_" + \
    "L" + str(args.n_labeled_0) + "_" + \
    "L" + str(args.n_labeled_delta) + "x" + str(args.n_labeled_sets) + "_" + \
    "U" + str(args.n_unlabeled) + "_" + \
    "V" + str(args.n_val) + "_" + \
    "T" + str(args.n_test) + "_" + \
    "seed" + str(args.seed) + \
    ".pickle"

with open(path_file, "wb") as f:
    dump(dict_data, f)

