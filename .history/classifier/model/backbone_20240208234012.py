import torchvision
import torch
from torch.nn import nn


def get_model(model_name, device):

    models = dict(
        resnet18 = models.resnet18(pretrained=True),
        alexnet = models.alexnet(pretrained=True),
        vgg16 = models.vgg16(pretrained=True),
        efficientnet_b0 = models.efficientnet_b0(pretrained=True),
        efficientnet_b1 = models.efficientnet_b1(pretrained=True)
    )


    model_conv = models[model_name]
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    
    return model_conv