import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os

import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.models import resnext50_32x4d, vgg16, resnet18, resnet50

from PIL import Image, ImageFilter, ImageDraw
from skimage.util import random_noise

import os
import random

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import Subset
import pandas as pd
import matplotlib.pyplot as plt

def train(model, train_loader, val_loader, optimizer, lr_scheduler, criterion, device, num_epochs, save_dir, model_name):
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} (Train)')
        for inputs, labels in train_bar:
            for ii in range(model.num_models):
                inputs[ii] = inputs[ii].to(device)
            labels = labels.to(torch.float).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels, device)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            #_, predicted = torch.max(outputs.data, 1)
            # Convert logits to probabilities using the sigmoid function
            probabilities = [torch.sigmoid(output) for output in outputs]

            # Define a threshold (e.g., 0.5)
            threshold = 0.5

            # Apply thresholding to get binary predictions
            binary_predictions = [(prob > threshold).type(torch.int) for prob in probabilities]

            # Compute the majority vote prediction
            majority_predictions = torch.sum(torch.stack(binary_predictions, dim=0).squeeze(dim=2), dim=0) >= (len(outputs) // 2 + 1)

            total_train += labels.size(0)
            correct_train += (majority_predictions == labels).sum().item()
            train_bar.set_postfix(loss=train_loss / (train_bar.n + 1), accuracy=correct_train / total_train)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} (Validation)')
        with torch.no_grad():
            for inputs, labels in val_bar:
                for ii in range(model.num_models):
                    inputs[ii] = inputs[ii].to(device)
                labels = labels.to(torch.float).to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels, device)

                val_loss += loss.item()

                #_, predicted = torch.max(outputs.data, 1)


                # Convert logits to probabilities using the sigmoid function
                probabilities = [torch.sigmoid(output) for output in outputs]

                # Define a threshold (e.g., 0.5)
                threshold = 0.5

                # Apply thresholding to get binary predictions
                binary_predictions = [(prob > threshold).type(torch.int) for prob in probabilities]

                # Compute the majority vote prediction
                majority_predictions = torch.sum(torch.stack(binary_predictions, dim=0).squeeze(dim=2), dim=0) >= (len(outputs) // 2 + 1)

                total_val += labels.size(0)
                correct_val += (majority_predictions == labels).sum().item()

                val_bar.set_postfix(loss=val_loss / (val_bar.n + 1), accuracy=correct_val / total_val)

        # Adjust learning rate
        lr_scheduler.step()

        # Save the model if it has the best validation accuracy
        if correct_val > best_val_acc:
            best_val_acc = correct_val
            best_epoch = epoch

            # Save the model checkpoint
            checkpoint_path = os.path.join(save_dir, f'{model_name}_best2.pth')
            torch.save(model.state_dict(), checkpoint_path)

    print(f'Best model found at epoch {best_epoch + 1} with validation accuracy of {best_val_acc / total_val * 100:.2f}%')
