
import torchvision 
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
from model.backbone import get_model
from classifier.utils.train import train_model

from datasets.load_data import get_dataloader

from utils.utils import set_seed, init_new_run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type = str, default = "binary classifier")
    parser.add_argument("--job_type", type = str, default = "train")
    
    
    parser.add_argument("--model_name", type = str, default="resnet18")
    parser.add_argument("--lr", type = float, default=3e-4 )
    parser.add_argument("--epochs", type = int, default=30 )
    parser.add_argument("--batch_size", type = int, default=16)
    config = parser.parse_args()
    
    
    
    set_seed()
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    dataloaders = get_dataloader("./data", batch_size=config.batch_size)

    model_conv = get_model(config.model_name)
    model_conv = model_conv.to(device)



    # loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=config.lr)


    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    
    
    run = init_new_run(name = "fine-tune CNN", job_type="training")
    
    
    model_conv = train_model(run, model_conv, config.model_name, criterion, optimizer_conv,
                            exp_lr_scheduler, num_epochs=config.epoch)
if __name__ == 'main':
    main()