import torch
import torch.nn as nn
from torchvision import models


def create_model():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        7
    )

    return model
