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

#data is stored in format ./Data/{lat}_{long}.tif
dataPath = "./Data"
data_transform = transforms.Compose([
  transforms.RandomSizedCrop(224),
  transforms.ToTensor()
  #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = datasets.ImageFolder(root=dataPath, transform=data_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

for i_batch, (sample_batched, idk) in enumerate(dataloader):
  # observe 4th batch and stop.
  if (i_batch == 3):
    print(idk)
    print(sample_batched)
    print(tf.shape(sample_batched[0]))
    plt.figure()
    plt.imshow(sample_batched)
    plt.axis('off')
    plt.ioff()
    plt.show()
    break