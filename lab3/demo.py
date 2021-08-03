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
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable

print(torch.__version__)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

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
            transforms.Resize(512),
            transforms.RandomRotation(360),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        s = preprocess(img)

        return s, self.label[index]


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

class ResNet18(nn.Module):
    def __init__(self,pretrained_type):
        super(ResNet18, self).__init__()
        self.name = 'ResNet18'
        self.pretrained_model = models.resnet18(pretrained = pretrained_type)
        self.classify = nn.Sequential(
            nn.Linear(in_features = 1000, out_features = 5, bias = True)
        )

    def forward(self, x):
        out = self.pretrained_model(x)
        out = self.classify(out)
        return out

config = {
    'Batch_size' : 4,
    'Epochs' : 10,
    'Optimizer' : 'SGD',
    'Optim_hparas':{
        'lr' : 0.001,
        'momentum' : 0.9,
        'weight_decay' : 5e-4
    },
    'Loss_function' : torch.nn.CrossEntropyLoss()
}
    
train_loader, test_loader = prep_dataloader('data/',config['Batch_size'])

model = ResNet18(True)
model.load_state_dict(torch.load('save/ResNet18_maxacc{}'.format('82'),map_location=torch.device('cpu')))
model.cuda() if torch.cuda.is_available() else model.cpu()


model.eval()
test_accuracy = 0
test_loss = 0
print('+---------------------Demo---------------------+')
for xx,yy in tqdm(test_loader):
    xx, testlabel = xx.to(device), yy.to(device)
    testpred = model(xx)
    sss = torch.max(testpred,1)[1]
    test_accuracy += sss.eq(testlabel).sum().item()
    loss2 = config['Loss_function'](testpred, testlabel)
    test_loss += loss2.item()
test_accuracy = test_accuracy*100./7025
print('accuracy = {}'.format(test_accuracy))
    
