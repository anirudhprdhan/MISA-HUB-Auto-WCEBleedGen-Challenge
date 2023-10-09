import torch
import torch.nn as nn
import torch.optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torchvision.models import resnext50_32x4d, vgg16, resnet18, resnet50



from tqdm import tqdm

from PIL import Image, ImageFilter, ImageDraw
from skimage.util import random_noise


import os
import random

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import Subset
import pandas as pd
import matplotlib.pyplot as plt

class EnsembleModel(nn.Module):
    def __init__(self, num_classes=1, num_models=5):
        super(EnsembleModel, self).__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        # self.models = nn.ModuleList([self.create_resnet18_binary() for _ in range(num_models)])
        self.models = nn.ModuleList([
            self.create_vgg(),
            self.create_mobilenet(),
            self.create_resnet18()
        ])

    def create_vgg(self):
        # Create and return a VGG16 model
        vgg = vgg16(weights=None)
        in_features = vgg.classifier[6].in_features
        vgg.classifier[6] = nn.Linear(in_features, self.num_classes)  # Modify the last fully connected layer
        nn.init.xavier_normal_(vgg.classifier[6].weight)
        return vgg

    def create_resnet18(self):
        # Create and return a ResNet-18 model
        res18 = resnet18(weights=None)
        in_features = res18.fc.in_features
        res18.fc = nn.Linear(in_features, self.num_classes)
        nn.init.xavier_normal_(res18.fc.weight)
        return res18
    def create_mobilenet(self):
        # Create and return a MobileNetV2 model
        mob = models.mobilenet_v2(weights=None)
        in_features = mob.classifier[1].in_features
        mob.classifier[1] = nn.Linear(in_features, self.num_classes)  # Modify the last fully connected layer
        nn.init.xavier_normal_(mob.classifier[1].weight)
        return mob

    def forward(self, x):
        # Forward pass through each model in the ensemble
        outputs = [model(x[i]) for i, model in enumerate(self.models)]
        return outputs
