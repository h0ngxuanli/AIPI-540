import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import wandb
import argparse
import random

cudnn.benchmark = True


parser = argparse.ArgumentParser()
parser.add_argument("--name", type = str, default = "binary classifier")
parser.add_argument("--job_type", type = str, default = "train")
parser.add_argument()
parser.add_argument()

config = parser.parse_args()