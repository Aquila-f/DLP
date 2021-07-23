import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import random
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torchsummary import summary
print(torch.__version__)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def read_bci_data(): 
    
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
    return train_data, train_label, test_data, test_label

def prep_dataloader(batch_size):
    train_data, train_label, test_data, test_label = read_bci_data()
    train_dataset = Data.TensorDataset(torch.from_numpy(train_data),torch.from_numpy(train_label))
    train_loader = Data.DataLoader(
        train_dataset,
        batch_size = config['Batch_size'],
        shuffle = True,
        num_workers = 4
    )

    test_dataset = Data.TensorDataset(torch.from_numpy(test_data),torch.from_numpy(test_label))
    test_loader = Data.DataLoader(
        test_dataset,
        batch_size = config['Batch_size'],
        shuffle = False,
        num_workers = 4
    )
    return train_loader,test_loader

def activation_funcchoose(act_func):
    if act_func == 'ReLU':
        return nn.ReLU()
    elif act_func == 'LeakyReLU':
        return nn.LeakyReLU()
    return nn.ELU()

def calwithlabel(test_loadeer,model,lossfunc):
    test_loss = 0
    test_accuracy = 0
    
    for x, y in test_loadeer:
        x, label = x.to(device ,dtype = torch.float), y.to(device ,dtype = torch.long)
        pred = model(x)
        test_accuracy += torch.max(pred,1)[1].eq(label).sum().item()
        test_loss += lossfunc(pred,label)
        
    test_accuracy = test_accuracy*100./1080
    return test_loss, test_accuracy
    

class eegNet(nn.Module):
    def __init__(self,act_func):
        self.act_funct = act_func
        super(eegNet,self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1,1), padding=(0,25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation_funcchoose(self.act_funct),
#             nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size = (1,4), stride=(1,4), padding=0),
            nn.Dropout(0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,15), stride=(1,1), padding=(0,7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation_funcchoose(self.act_funct),
#             nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size = (1,8), stride=(1,8), padding=0),
            nn.Dropout(0.25),
            nn.Flatten()
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )
        
    def forward(self,x):
        out = self.firstconv(x)
        out = self.depthwiseConv(out)
        out = self.separableConv(out)
        out = self.classify(out)
        return out


config = {
    'Epochs' : 150,
    'Batch_size' : 64,
    'Optimizer' : 'Adam',
    'Optim_hparas':{
        'lr' : 1e-2
    },
    'Loss_function' : torch.nn.CrossEntropyLoss(),
    'print_step': 10,
    'activation_function' : 'ELU'
}
print('ELU')

train_loader,test_loader = prep_dataloader(config['Batch_size'])
train_accuracy_list = []
train_loss_list = []
test_accuracy_list = []
test_loss_list = []

model = eegNet(config['activation_function'])
model.cuda()
epoch = config['Epochs']
# optimizer = config['Optimizer'](model.parameters(), lr = config['Learning_rate'], )
optimizer = getattr(torch.optim, config['Optimizer'])(model.parameters(), **config['Optim_hparas'])
printstep = config['print_step']

for i in range(1,config['Epochs']+1):
    train_loss = 0
    train_accuracy = 0
    
    for x, y in train_loader:
        optimizer.zero_grad()
        x, label = x.to(device ,dtype = torch.float), y.to(device ,dtype = torch.long)
        pred = model(x)
        train_accuracy += torch.max(pred,1)[1].eq(label).sum().item()
        loss = config['Loss_function'](pred,label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_accuracy = train_accuracy*100./1080
    test_loss,test_accuracy = calwithlabel(test_loader,model,config['Loss_function'])
    
    test_accuracy_list.append(test_accuracy)
    test_loss_list.append(test_loss)
    train_accuracy_list.append(train_accuracy)
    train_loss_list.append(train_loss)
    
    
    if i % printstep == 0:
        print('epoch : {}, loss : {}, accurancy : {:.2f}'.format(i,test_loss,test_accuracy))
        
print(train_accuracy_list)
print(train_loss_list)
print(test_accuracy_list)
print(test_loss_list)
