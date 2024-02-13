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
from tqdm import tqdm
import wandb
import argparse
import random
from utils.utils import set_seed, init_new_run, create_dataset_artifact, create_model_artifact, get_metrics
from pathlib import Path
cudnn.benchmark = True





def train_model(run, model, model_name, dataloaders, criterion, optimizer, scheduler, device, epochs):

    best_auc = 0.0

    for epoch in tqdm(epochs, desc='Processing'):
        
        start_time = time.time()
        
        
        # Each epoch condcut training and validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0



            true_labels = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # results
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            
            
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[[phase]].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[[phase]].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # load latest model to artifact
            if phase == 'val' and epoch_acc > best_acc:
                
                best_acc = epoch_acc
                
                best_model_path = "./model" / "best_model_param.pth"
                
                torch.save(model.state_dict(), best_model_path)
                
                create_model_artifact(best_model_path, run, model_name)

        print(print(f'Epoch {epoch}/{epochs - 1}''-' * 10))

    time_elapsed = time.time() - since
    
    
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
        
        
        
    return model