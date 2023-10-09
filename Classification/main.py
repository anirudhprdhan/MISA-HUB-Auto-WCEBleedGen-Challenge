import numpy as np

from torch.utils.data import Subset

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

from dataset import WCEClassDataset, WCEClassSubsetDataset
from data_augmentation import WCEImageTransforms
from model import EnsembleModel
from train import train

def ensemble_loss(outputs, targets, device):
    # BCELoss for each model in the ensemble
    bce_loss = nn.BCEWithLogitsLoss().to(device)

    # Compute individual losses and sum them up
    losses = [bce_loss(F.sigmoid(output), targets.unsqueeze(1)) for output in outputs]
    total_loss = sum(losses)

    return total_loss

def main():
    root_dir = '/kaggle/input/dataset/upload'
    # root_dir = '../datasets/WCEBleedGen'
    # val_dir = '/content/drive/MyDrive/Challenge_bleed/Test Dataset 1'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_models= 5
    dataset = WCEClassDataset(root_dir=root_dir, num_models=num_models) # loading the training Dataset
    # val_dataset = WCEValDataset(root_dir=val_dir, num_models=num_models ) # loading the test Dataset

    # number of epochs
    num_epochs = 50


    batch_size = 16
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    save_dir = '/kaggle/working/'
    # save_dir = './checkpoints'

    model_name = 'WCE_class'

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

#     val_indices= list(range(len(val_dataset)))

    # Creating PT data samplers and loaders:
    rotation_degrees = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
    blur_parameters = [(11, 9), (9, 7), (7, 5), (5, 3), (3, 1)]
    train_transform = WCEImageTransforms(rotation_degrees, blur_parameters)
    valid_transform = torchvision.transforms.ToTensor()
    train_dataset= WCEClassSubsetDataset(dataset, train_indices, train_transform)
    valid_dataset = WCEClassSubsetDataset(dataset, val_indices, valid_transform)
    # valid_dataset = WCEValidDataset('/content/drive/MyDrive/Test Dataset 1', valid_transform)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,num_workers=2)
    validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    model = EnsembleModel(num_models=5)
    model.to(device)

    lr = 0.01
    weight_decay = 1.0e-4
    # Create the SGD optimizer with momentum and weight decay
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    criterion = ensemble_loss

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)

    train(model, train_loader, validation_loader, optimizer, lr_scheduler, criterion, device, num_epochs, save_dir, model_name)

if __name__ == '__main__':
    main()
