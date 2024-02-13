
import torch
import random
import os
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")



def get_metrics(preds, true):
    

def init_new_run(name,job_type):
    run = wandb.init(project="aipi-540",job_type=job_type,name=name)
    return run

def create_dataset_artifact(run,name):
    artifact = wandb.Artifact(name,type='dataset')
    artifact.add_dir('data/train')
    artifact.add_dir('data/val')
    artifact.add_dir('data/test')
    
    # This will create an artifact in W&B if it does not exist yet
    run.use_artifact(artifact)


def create_model_artifact(path,run,name):
    artifact = wandb.Artifact(name,type='model')
    artifact.add_file(path)
    run.log_artifact(artifact)
