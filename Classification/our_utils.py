import torch
import torch.nn as nn
import torch.optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnext50_32x4d, vgg16, resnet18, resnet50



from tqdm import tqdm

from PIL import Image, ImageFilter, ImageDraw
from skimage.util import random_noise

import random


from torch.utils.data import Dataset
from torch.utils.data import Subset
import pandas as pd
import matplotlib.pyplot as plt



import os
import numpy as np
from sklearn.cluster import KMeans

# Function to perform minimum variance quantization (MVQ) on a single image and save the quantized version
def quantize_image(input_image_path, output_image_path, num_colors=24):
    # Load the image
    image = Image.open(input_image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Flatten the 2D array to a 1D array of RGB values
    pixels = image_array.reshape((-1, 3))

    # Apply K-Means clustering to quantize the colors
    kmeans = KMeans(n_clusters=num_colors, n_init='auto', random_state=0).fit(pixels)
    quantized_colors = kmeans.cluster_centers_.astype(int)

    # Replace each pixel with its nearest quantized color
    quantized_pixels = quantized_colors[kmeans.labels_]

    # Reshape the quantized pixels back to the original image shape
    quantized_image_array = quantized_pixels.reshape(image_array.shape)

    # Create a new image from the quantized pixel values
    quantized_image = Image.fromarray(quantized_image_array.astype(np.uint8))

    # Save the quantized image
    quantized_image.save(output_image_path)

if __name__ == '__main__':

    # Specify the folder containing input images and the folder to save quantized images
    input_folder = '../datasets/WCEBleedGen/non-bleeding/images'
    output_folder = '../datasets/WCEBleedGen/non-bleeding/quantized_images/'

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    input_files = os.listdir(input_folder)

    # Iterate through each image file and quantize it
    for ii, input_file in enumerate(input_files):
        if input_file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_image_path = os.path.join(input_folder, input_file)
            output_image_path = os.path.join(output_folder, input_file)
            quantize_image(input_image_path, output_image_path)
            print(f"Quantized: {ii}, {input_file}")

    print("Quantization complete.")

