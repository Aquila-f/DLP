
import pandas as pd
import numpy as np
import math
from PIL import Image
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms,models
from matplotlib import pyplot as plt
from tqdm import tqdm
class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):

        path = '{}{}.jpeg'.format(self.root, self.img_name[index])
        img = Image.open(path)
        
        preprocess = transforms.Compose([
        #     transforms.Resize(512),
            transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        s = preprocess(img)
        s /= 255        
        return s, self.label[index]

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

def prep_dataloader(root, Batch_size):
    train_dataset = RetinopathyLoader(root, 'train')
    train_loader = data.DataLoader(
        train_dataset,
        batch_size = Batch_size,
        shuffle = True,
        num_workers = 4
    )
    
    test_dataset = RetinopathyLoader(root, 'test')
    test_loader = data.DataLoader(
        test_dataset,
        batch_size = Batch_size,
        shuffle = False,
        num_workers = 4
    )
    return train_loader, test_loader

train_loader, test_loader = prep_dataloader('data/',4)

for x,y in test_loader:
    print(x.shape)

for x,y in train_loader:
    print('d')
