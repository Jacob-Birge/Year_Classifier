import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import *
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint

def imSave(inp, filename):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imsave(filename+".png", inp)
    plt.pause(0.001)  # pause a bit so that plots are updated

class LightningResnet18(pl.LightningModule):
    def __init__(self):
        super(LightningResnet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 9)
        self.lossFunc = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=0.001)
    
    def loss(self, logits, labels):
        return self.lossFunc(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('val_loss', loss)

class LightningDataModule(pl.LightningDataModule):
    def setup(self, stage=None):
        #data is stored in format ./Data/{lat}_{long}.tif
        dataPath = "./Data"
        #define transforms
        data_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor()
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        #pull in dataset
        dataset = datasets.ImageFolder(root=dataPath, transform=data_transform)
        self.class_names = dataset.classes
        self.numClasses = len(self.class_names)
        #split train and val set
        numInTrain = int(len(dataset)*.8)
        splitSet = torch.utils.data.random_split(dataset, [numInTrain, len(dataset)-numInTrain])
        self.trainSet = splitSet[0]
        self.valSet = splitSet[1]
    
    def train_dataloader(self):
        return DataLoader(self.trainSet, batch_size=4, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valSet, batch_size=4, shuffle=True, num_workers=8)

dataModule = LightningDataModule()
model = LightningResnet18.load_from_checkpoint("~/SMART_Project/lightning_logs/version_9/checkpoints/epoch=10-step=102365.ckpt")
model.eval()

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix=''
)

trainer = pl.Trainer(gpus=[0], max_epochs=25, checkpoint_callback=checkpoint_callback)
trainer.fit(model, dataModule)