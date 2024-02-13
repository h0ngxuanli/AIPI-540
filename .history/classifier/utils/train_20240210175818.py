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
from tqdm import tqdm, trange
import wandb
import argparse
import random
from utils.utils import set_seed, init_new_run, create_dataset_artifact, create_model_artifact, get_metrics
from pathlib import Path
from datetime import datetime
cudnn.benchmark = True





def train_model(run, model, model_name, dataloaders, criterion, optimizer, scheduler, device, epochs):

    best_auc = 0.0

    for epoch in trange(epochs, desc='Processing'):
        
        start_time = time.time()
        running_loss = 0.0
        metrics = {}
        
        # Each epoch condcut training and validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

        
            y_trues = torch.zeros((len(dataloaders[phase].dataset), ), dtype=torch.float32, device=device)
            y_preds = torch.zeros((len(dataloaders[phase].dataset), ), dtype=torch.float32, device=device)
            
            
            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
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
                
                # print(labels)
                # print(labels.shape)

                y_trues[i*dataloaders[phase].batch_size:(i*dataloaders[phase].batch_size + inputs.size(0))] = labels
                y_preds[i*dataloaders[phase].batch_size:(i*dataloaders[phase].batch_size + inputs.size(0))] = preds
                
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[[phase]].dataset)
            epoch_acc, epoch_auc, epoch_f1, epoch_recall, epoch_precision = get_metrics(y_trues, y_preds)
            
            metrics[phase] = dict(
                zip(["Epoch", phase + " Acc", phase + " Auc", phase + " F1-score", phase + " Recall", phase + " Precision"], 
                    [epoch_loss, epoch_acc, epoch_auc, epoch_f1, epoch_recall, epoch_precision])
            )
            
            

            print('Epoch {:03}: {} | Loss: {:.3f} | Acc: {:.3f} | Auc: {:.3f} | F1-score: {:.3f} | Training time: {}'.format(
                        epoch + 1, 
                        phase, 
                        epoch_loss, 
                        epoch_acc, 
                        epoch_auc, 
                        epoch_f1, 
                        str(datetime.timedelta(seconds=time.time() - start_time))[:7]))

            # load latest model to artifact
            if phase == 'val' and epoch_auc > best_auc:
                
                best_auc = epoch_auc
                
                best_model_path = Path("./model/checkpoints") / (model_name + ".pth")
                
                torch.save(model.state_dict(), best_model_path)
                
                create_model_artifact(best_model_path, run, model_name)

        wandb.log({**metrics["train"], **metrics["val"]})
        wandb.log({"Epoch Time": str(datetime.timedelta(seconds=time.time() - start_time))[:7]})
        

    time_elapsed = time.time() - start_time
    
    wandb.log({"Total Time": time_elapsed})
    
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_auc:4f}')
        
    return model