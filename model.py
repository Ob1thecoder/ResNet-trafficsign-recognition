# model_def.py
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_resnet18(num_classes=43, pretrained=True):
    if pretrained:
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        m = resnet18(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, num_classes)
    return m
