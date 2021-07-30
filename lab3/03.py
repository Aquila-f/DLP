import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms,models

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
        #     transforms.Resize(512),
            transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        s = preprocess(img)
        s /= 255

        
        return s, self.label[index]

def prep_dataloader(root, batch_size):
    train_dataset = RetinopathyLoader(root, 'train')
    train_loader = data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True
    )
    
    test_dataset = RetinopathyLoader(root, 'test')
    test_loader = data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = False
    )
    return train_loader, test_loader

class ResNet18(nn.Module):
    def __init__(self,pretrained_type):
        super(ResNet18, self).__init__()
        
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
    'Learning_rate' : 0.001,
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
model.cuda() if torch.cuda.is_available() else model.cpu()
optimizer = getattr(torch.optim, config['Optimizer'])(model.parameters(), **config['Optim_hparas'])

for epoch in range(1,config['Epochs']+1):
    
    train_loss = 0
    train_accuracy = 0
    test_loss = 0
    test_accuracy = 0
    i = 0
    
    model.train()
    for x,y in train_loader:
        optimizer.zero_grad()
#         x, label = x.to(device ,dtype = torch.float), y.to(device ,dtype = torch.long)
        pred = model(x)
        print(torch.max(pred,1)[1])
        print(y)
        print(torch.max(pred,1)[1].eq(y).sum().item())
        
        train_accuracy += torch.max(pred,1)[1].eq(label).sum().item()
        loss = config['Loss_function'](pred, label)
        train_loss += loss.item()
        print(loss)
        loss.backward()
        optimizer.step()
        i+=1
        if i%100==0: print(i)
    print('train - epoch : {}, loss : {}, accurancy : {:.2f}'.format(i,train_loss,train_accuracy))
    print(i)
        
        
    
        



