from PIL import Image
from os import listdir
from os.path import isfile, join
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import time
import copy
import os
import tensorflow as tf

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#data is stored in format ./Data/{lat}_{long}.tif
dataPath = "./Data"
data_transform = transforms.Compose([
  transforms.RandomSizedCrop(224),
  transforms.ToTensor()
  #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = datasets.ImageFolder(root=dataPath, transform=data_transform)
class_names = dataset.classes

numInTrain = int(len(dataset)*.8)

splitSet = torch.utils.data.random_split(dataset, [numInTrain, len(dataset)-numInTrain])
image_datasets = {x: splitSet[i]
                  for i, x in enumerate(['train', 'val'])}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])