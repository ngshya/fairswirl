# FairSwiRL
Fair Semi-supervised with Representation Learning


## Python Environment

In order to recreate the Python environment, use the `package-list.txt` file:

```bash
conda create --name <env> --file package-list.txt
```

then, activate the env. 


## Data splits generation

The experiments are run with different splits generated from the same dataset. For example, in order to generate a split from the synthetic dataset:

```bash
python scripts/data_generation.py --dataset synthetic --seed 1101
```

In our experiments, we have used the seeds from 1101 to 1110. 


## Experiments with Weights & Biases

The experiments are performed by using [Weights & Biases](https://wandb.ai/). Inside the folder `sweeps` there are sample YAML files for each experiment presented in the paper. Just remember to adjust the dataset and the splits names inside the YAML file accordingly. 

To run the hyperparameters search, use the following YAML files: 
- `hyp_search_fairswirl.yaml` (FairSwiRL)
- `hyp_search_fesf.yaml` (FESF)
- `hyp_search_fairssl.yaml` (FairSSL)

To run the experiments, use one of the following sample YAML files with the hyperparameters found in the previous step:
- `inc_labels_fairswirl_100_2000.yaml` (FairSwiRL+RF, FairSwiRL+RF WOUD, DD+RF, number of labeled instances from 100 to 2000)
- `inc_labels_fairswirl_other_classifiers_100_2000.yaml`(FairSwiRL+RF, FairSwiRL+KNN, FairSwiRL+LR, FairSwiRL+SVC, number of labeled instances from 100 to 2000)
- `inc_labels_competitors_100_2000.yaml` (FESF, FairSSL, number of labeled instances from 100 to 2000)
- `inc_labels_pd_rf.yaml` (PD+RF, number of labeled instances from 100 to 2000)
- `inc_labels_fairswirl_10_100.yaml` (FairSwiRL+RF, FairSwiRL+RF WOUD, DD+RF, number of labeled instances from 10 to 100)
- `inc_labels_competitors_10_100.yaml` (FESF, FairSSL, number of labeled instances from 10 to 100)


