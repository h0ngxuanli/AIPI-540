
import torch
import random
import os
import wandb
import numpy as np


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


def init_new_run(name,job):
    run = wandb.init(project="aipi-540",job_type=job,name=name)
    return run

def create_dataset_artifact(run,name):
    artifact = wandb.Artifact(name,type='dataset')
    artifact.add_dir('data/train')
    artifact.add_dir('data/val')
    artifact.add_dir('data/test')
    
    # 
    run.use_artifact(artifact)


def create_model_artifact(path,run,name):
    artifact = wandb.Artifact(name,type='model')
    artifact.add_file(path)
    run.log_artifact(artifact)
