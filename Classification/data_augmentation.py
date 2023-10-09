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

from torch.utils.data import Dataset
from torch.utils.data import Subset
import pandas as pd
import matplotlib.pyplot as plt

import os
import random
import numpy as np

class WCEImageTransforms:
    def __init__(self, rotation_degrees, blur_parameters):
        self.rotation_degrees = rotation_degrees
        self.blur_parameters = blur_parameters

    def apply_poisson_noise(self, image):
        image_np = np.array(image)
        noisy_image = random_noise(image_np, mode='poisson', rng=None) * 255
        return Image.fromarray(noisy_image.astype(np.uint8))

    def __call__(self, img):
        # Randomly decide whether to rotate the image
        should_rotate = random.choice([True, False])
        if should_rotate:
            # Randomly select a degree from the set and rotate the image
            random_degree = random.choice(self.rotation_degrees)
            img = img.rotate(random_degree, expand=True)

        # Randomly decide whether to convert to YCbCr
        should_convert = random.choice([True, False])
        # if should_convert:
        #     # Convert to YCbCr color space
        #     img = np.array(img.convert('YCbCr'))

        #     # Choose a random scaling
        #     random_scaling_factor = random.random()

        #     # Apply the scaling factor to the Y channel
        #     img[:, :, 0] = (img[:, :, 0] * random_scaling_factor).clip(0, 255).astype(np.uint8)

        #     # Convert the modified YCbCr image back to RGB
        #     img = Image.fromarray(img, mode='YCbCr').convert('RGB')

        if should_convert:
            # Convert the image to the HSV color space
            img_hsv = img.convert('HSV')

            # Choose a random scaling factor for the Value (V) channel
            random_scaling_factor = random.uniform(0.5, 2.0)  # You can adjust the scaling factor range as needed

            # Convert the HSV image to a NumPy array
            img_array = np.array(img_hsv)

            # Apply the scaling factor to the Value (V) channel
            img_array[:, :, 2] = (img_array[:, :, 2] * random_scaling_factor).clip(0, 255).astype(np.uint8)

            # Convert the modified HSV image back to RGB
            img = Image.fromarray(img_array, mode='HSV').convert('RGB')

        # # Randomly decide whether to apply Gaussian blur
        # should_blur = random.choice([True, False])
        # if should_blur:
        #     # Randomly select blur parameters (w, sigma) from the set
        #     random_w, random_sigma = random.choice(self.blur_parameters)

        #     # Apply Gaussian blur to the image
        #     img = img.filter(ImageFilter.GaussianBlur(radius=random_sigma))

        # # Randomly decide whether to add Poisson noise
        # should_add_poisson_noise = random.choice([True, False])
        # if should_add_poisson_noise:
        #     # Apply Poisson noise to the image
        #     img = self.apply_poisson_noise(img)

        # Convert the PIL image to a PyTorch tensor
        trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        img_tensor = trans(img)

        return img_tensor