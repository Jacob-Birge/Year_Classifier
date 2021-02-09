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
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

#data is stored in format ./Data/{lat}_{long}.tif
dataPath = "./Data"
data_transform = transforms.Compose([
  transforms.RandomSizedCrop(224),
  transforms.ToTensor()
  #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = datasets.ImageFolder(root=dataPath, transform=data_transform)
class_names = dataset.classes
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

for i_batch, (inputs, classes) in enumerate(dataloader):
  # observe 4th batch and stop.
  if (i_batch == 3):
    print(classes)
    print(tf.shape(inputs))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])