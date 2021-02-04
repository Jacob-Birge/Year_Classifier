from PIL import Image
import pytorch_lightning as pl
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

import matplotlib.pyplot as plt
import time
import copy
import os

#data is stored in format ./Data/{x_bound}/{y_bound}/{file}.tif
dataPath = "./Data"
sourcePathsByYear = {}
yearDirs = [d for d in listdir(dataPath) if not isfile(join(dataPath, d))]
for yearDir in yearDirs:
  sourcePathsByYear[yearDir] = []
  yearDirFull = join(dataPath, yearDir)
  xDirs = [join(yearDirFull, d) for d in listdir(yearDirFull) if not isfile(join(yearDirFull, d))]
  values = range(len(xDirs))
  with tqdm(total=len(values), file=sys.stdout) as pbar:
    i=1
    for xDir in xDirs:
      yDirs = [join(xDir, d) for d in listdir(xDir) if not isfile(join(xDir, d))]
      for yDir in yDirs:
        files = [join(yDir, f) for f in listdir(yDir) if isfile(join(yDir, f))]
        sourcePathsByYear[yearDir] += files
      pbar.set_description('processed: %d' % (i))
      i+=1
      pbar.update(1)
  print(len(sourcePathsByYear[yearDir]))

im = Image.open('./Data/2020/440000/4974000/440000.0_4974000.0.tif')
train_data = np.array( [np.asarray(Image.open(sourcePathsByYear["2020"][i])) for i in range(len(sourcePathsByYear["2020"]))] )
print(train_data.shape)