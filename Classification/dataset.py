import os
import torch
from torch.utils.data import Dataset

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

import random

import numpy as np

from torch.utils.data import Subset
import pandas as pd
import matplotlib.pyplot as plt


class WCEClassDataset(Dataset):
    def __init__(self, root_dir, num_models=5):
        super().__init__()
        self.root_dir = root_dir
        self.num_models = 5


        # List of subfolders (class names)
        self.classes = ['/kaggle/input/wce-quantized-img/quantized_images', '/kaggle/input/quantized-nonbleed-images/quantized_images']

        # Initialize lists to hold image paths and labels
        self.image_paths = []
        self.labels = []

        # Load image paths and labels
        for class_idx, class_dir in enumerate(self.classes):
            image_files = os.listdir(class_dir)
            for image_file in image_files:
                image_path = os.path.join(class_dir, image_file)
                self.image_paths.append(image_path)
                self.labels.append(class_idx) # bleeding images given label 0 and non-bleeding 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image using PIL
        image = Image.open(image_path)

        return image, label


class WCEClassSubsetDataset(Dataset):
    def __init__(self, original_dataset, subset_indices, transform=None):
        self.original_dataset = original_dataset
        self.subset_indices = subset_indices
        self.transform = transform

    def __len__(self):
        return len(self.subset_indices)

    def __getitem__(self, idx):
        # Get an item from the subset
        image, label = self.original_dataset[self.subset_indices[idx]]

        # Apply the transform if it is provided
        if self.transform:
            images = []
            for i in range(self.original_dataset.num_models):
                images.append(self.transform(image))

            return images, label

        return image, label

class WCEDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(os.path.join(root_dir, 'Images')) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.annotation_files = [f for f in os.listdir(os.path.join(root_dir, 'Bounding boxes', 'YOLO_TXT')) if f.endswith('.txt')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'Images', self.image_files[idx])
        annotation_name = os.path.join(self.root_dir, 'Bounding boxes', 'YOLO_TXT', self.image_files[idx][:-4]+'.txt')

        # Load image
        image = Image.open(img_name).convert('RGB')

        # Load YOLO-style bounding box annotations
        targets = []
        with open(annotation_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                class_label = int(line[0])
                x_center, y_center, width, height = map(float, line[1:])
                targets.append({
                    'boxes': torch.tensor([x_center, y_center, width, height], dtype=torch.float32),
                    'labels': torch.tensor(class_label, dtype=torch.int64)
                })

        if self.transform:
            image = self.transform(image)

        return image, targets

    def save_image_with_boxes(self, idx, output_dir):
        img_name = os.path.join(self.root_dir, 'Images', self.image_files[idx])
        annotation_name = os.path.join(self.root_dir, 'Bounding boxes', 'YOLO_TXT', self.image_files[idx][:-4]+'.txt')

        image = Image.open(img_name).convert('RGB')
        draw = ImageDraw.Draw(image)

        with open(annotation_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                class_label = int(line[0])
                x_center, y_center, width, height = map(float, line[1:])
                x_center, y_center, width, height = x_center * image.width, y_center * image.height, width * image.width, height * image.height
                x1, y1, x2, y2 = x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2
                draw.rectangle([x1, y1, x2, y2], outline='red', width=1)
                #draw.text((x1, y1), f'Class {class_label}', fill='red')

        #print("Sairam", img_name, annotation_name, output_dir)
        image.save(os.path.join(output_dir, self.image_files[idx]))

class WCEValDataset(Dataset):
    def __init__(self, root_dir, num_models=5):
        super().__init__()
        self.root_dir = root_dir
        self.num_models= num_models


        # Initialize lists to hold image paths and labels
        self.image_paths = []
        self.labels = []

        # Load image paths and labels
        image_files = os.listdir(self.root_dir)
        for image_file in image_files:
            image_path = os.path.join(self.root_dir, image_file)
            self.image_paths.append(image_path)
            self.labels.append(0) # bleeding images given label 0 and non-bleeding 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image using PIL
        image = Image.open(image_path)

        return image, label
