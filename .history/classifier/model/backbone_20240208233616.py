import torchvision
import torch


model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)