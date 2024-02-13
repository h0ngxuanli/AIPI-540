from torchvision import models
import torchvision
import torch
import torch.nn as nn


def get_model(model_name):

    model_conv_dict = dict(
        resnet18 = models.resnet18(pretrained=True),
        alexnet = models.alexnet(pretrained=True),
        vgg16 = models.vgg16(pretrained=True),
        efficientnet_b0 = models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    )


    print(model_conv_dict)

    model_conv = model_conv_dict[model_name]
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    
    return model_conv