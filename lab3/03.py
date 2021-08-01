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
    
class ResNet50(nn.Module):
    def __init__(self,pretrained_type):
        super(ResNet50, self).__init__()
        self.name = 'ResNet50'
        self.pretrained_model = models.resnet50(pretrained = pretrained_type)
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
    'Epochs' : 5,
    'Optimizer' : 'SGD',
    'Optim_hparas':{
        'lr' : 0.001,
        'momentum' : 0.9,
        'weight_decay' : 5e-4
    },
    'Loss_function' : torch.nn.CrossEntropyLoss()
}


train_loader, test_loader = prep_dataloader('data/',config['Batch_size'])



df_acc = pd.DataFrame()
df_loss = pd.DataFrame()

for switch in [True]:
    
    train_accuracy_list = []
    train_loss_list = []
    test_accuracy_list = []
    test_loss_list = []
    test_max_acc = 0
    
    
    
    model = ResNet18(switch)
    model.cuda() if torch.cuda.is_available() else model.cpu()
    optimizer = getattr(torch.optim, config['Optimizer'])(model.parameters(), **config['Optim_hparas'])
    
    for epoch in range(1,config['Epochs']+1):
        train_loss = 0
        train_accuracy = 0
        test_loss = 0
        test_accuracy = 0
        

        model.train()
        for x,y in tqdm(train_loader):
            optimizer.zero_grad()
            x, label = x.to(device), y.to(device)
            pred = model(x)
            train_accuracy += torch.max(pred,1)[1].eq(label).sum().item()
            loss = config['Loss_function'](pred, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = train_loss/math.ceil(28099/config['Batch_size'])
        train_accuracy = train_accuracy*100./28099
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        print('train - epoch : {}, loss : {}, accurancy : {:.2f}'.format(epoch,train_loss,train_accuracy))

        model.eval()
        for xx,yy in tqdm(test_loader):
            xx, testlabel = xx.to(device), yy.to(device)
            testpred = model(xx)
            test_accuracy += torch.max(testpred,1)[1].eq(testlabel).sum().item()
            loss2 = config['Loss_function'](testpred, testlabel)
            test_loss += loss2.item()
        test_loss = test_loss/math.ceil(7025/config['Batch_size'])
        test_accuracy = test_accuracy*100./7025
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)
        
        if test_accuracy > test_max_acc: test_max_acc = test_accuracy
        if test_accuracy > 80 and test_accuracy == test_acc_max:
            torch.save(model.state_dict(),'{}_maxacc'.format(model.name))
            print(test_accuracy)


        print('test - epoch : {}, loss : {}, accurancy : {:.2f}'.format(epoch,test_loss,test_accuracy))
    if switch:
        df_acc['Test(with pretraining)'] = test_accuracy_list
        df_acc['Train(with pretraining)'] = train_accuracy_list
    else:
        df_acc['Test(w/o pretraining)'] = test_accuracy_list
        df_acc['Train(w/o pretraining)'] = train_accuracy_list

df_acc.index += 1
plt.figure(figsize=(9,6))
plt.plot(df_acc,'-o',markersize=3)
plt.grid()
plt.legend(df_acc.columns.values)
plt.title('Result Comparison({})'.format(model.name), fontsize=12)
plt.ylabel('Accuracy(%)')
plt.xlabel('Epochs')
plt.savefig('{}_acc.png'.format(model.name))



