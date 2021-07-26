import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
# import random
import pandas as pd
from matplotlib import pyplot as plt
import torch
# from torchsummary import summary
print(torch.__version__)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# torch.cuda.empty_cache()

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
    )

    test_dataset = Data.TensorDataset(torch.from_numpy(test_data),torch.from_numpy(test_label))
    test_loader = Data.DataLoader(
        test_dataset,
        batch_size = config['Batch_size'],
    )
    return train_loader,test_loader

def activation_funcchoose(act_func):
    if act_func == 'ReLU':
        return nn.ReLU()
    elif act_func == 'LeakyReLU':
        return nn.LeakyReLU()
    return nn.ELU(alpha=1.0)

# def calwithlabel(test_loadeer,model,lossfunc):
#     model.eval()
#     test_loss = 0
#     test_accuracy = 0
    
#     for x, y in test_loadeer:
#         x, label = x.to(device ,dtype = torch.float), y.to(device ,dtype = torch.long)
#         pred = model(x)
#         test_accuracy += torch.max(pred,1)[1].eq(label).sum().item()
#         test_loss += lossfunc(pred,label)
        
#     test_accuracy = test_accuracy*100./1080
#     return test_loss, test_accuracy
    

class eegNet(nn.Module):
    def __init__(self,act_func):
        self.name = 'EEGNet'
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
            nn.AvgPool2d(kernel_size = (1,4), stride=(1,4), padding=0),
#             nn.Dropout(0.25),
            nn.Dropout(0.5),
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,15), stride=(1,1), padding=(0,7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation_funcchoose(self.act_funct),
            nn.AvgPool2d(kernel_size = (1,8), stride=(1,8), padding=0),
#             nn.Dropout(0.25),
            nn.Dropout(0.5),
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

class DeepConvNet(nn.Module):
    def __init__(self,act_func):
        self.name = 'DeepConvNet'
        self.act_funct = act_func
        super(DeepConvNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1,5)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=(2,1)),
            nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
            activation_funcchoose(self.act_funct),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(0.75)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1,5)),
            nn.BatchNorm2d(50, eps=1e-5, momentum=0.1),
            activation_funcchoose(self.act_funct),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(0.75)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1,5)),
            nn.BatchNorm2d(100, eps=1e-5, momentum=0.1),
            activation_funcchoose(self.act_funct),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(0.5)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1,5)),
            nn.BatchNorm2d(200, eps=1e-5, momentum=0.1),
            activation_funcchoose(self.act_funct),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(0.5),
            nn.Flatten()
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=8600, out_features=2, bias=True)
        )
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.classify(out)
        return out

config = {
    'model' : [DeepConvNet],
    'Epochs' : 450,
    'Batch_size' : int(input('Batch_size : ')),
    'Optimizer' : str(input('optimizer : ')),
    'Optim_hparas':{
        'lr' : float(input('lr : '))
    },
    'Loss_function' : torch.nn.CrossEntropyLoss(),
    'print_step': 30,
    'activation_function' : ['ELU','ReLU','LeakyReLU']
}

train_loader,test_loader = prep_dataloader(config['Batch_size'])
epoch = config['Epochs']
# optimizer = config['Optimizer'](model.parameters(), lr = config['Learning_rate'], )
printstep = config['print_step']
df_max = pd.DataFrame(columns = {'ELU':0,'RelU':1,'LeakyReLU':2})

for modeltype in config['model']:
    
    dfacc = pd.DataFrame()
    dfloss = pd.DataFrame()
    test_acc_max_list = []
    train_acc_max_list = []
    
    for activation_function in config['activation_function']:

        train_accuracy_list = []
        train_loss_list = []
        test_accuracy_list = []
        test_loss_list = []
        
        test_acc_max = 0
        train_acc_max = 0

        model = modeltype(activation_function)
        
#         for param in model.parameters():
#             print(param.data)
#             break
#         torch.save(model.state_dict(),'save')
#         gg = modeltype(activation_function)
#         for param in gg.parameters():
#             print(param.data)
#             break
#         gg.load_state_dict(torch.load('save'))
#         for param in gg.parameters():
#             print(param.data)
#             break
        key = False
        
        print('{} , {}------------------------------'.format(model.name,activation_function))
        model.cuda()
        optimizer = getattr(torch.optim, config['Optimizer'])(model.parameters(), **config['Optim_hparas'])

        for i in range(1,config['Epochs']+1):
            train_loss = 0
            train_accuracy = 0
            test_loss = 0
            test_accuracy = 0
            
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                x, label = x.to(device ,dtype = torch.float), y.to(device ,dtype = torch.long)
                pred = model(x)
                train_accuracy += torch.max(pred,1)[1].eq(label).sum().item()
                loss = config['Loss_function'](pred,label)
                #flood = (loss-0.32).abs()+0.32
                #flood.backward()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                
            train_loss = train_loss/(1080./config['Batch_size'])
            train_accuracy = train_accuracy*100./1080
            
            
            model.eval()
            for xx, yy in test_loader:
                xx, testlabel = xx.to(device ,dtype = torch.float), yy.to(device ,dtype = torch.long)
                testpred = model(xx)
                test_accuracy += torch.max(testpred,1)[1].eq(testlabel).sum().item()
                loss2 = config['Loss_function'](testpred,testlabel)
                test_loss += loss2.item()
            test_accuracy = test_accuracy*100./1080
            test_loss = test_loss/(1080./config['Batch_size'])

            
            test_accuracy_list.append(test_accuracy)
            test_loss_list.append(test_loss)
            train_accuracy_list.append(train_accuracy)
            train_loss_list.append(train_loss)
            test_acc_max = test_accuracy if test_accuracy > test_acc_max else test_acc_max
            train_acc_max = train_accuracy if train_accuracy > train_acc_max else train_acc_max
            if test_accuracy > 87 and test_accuracy == test_acc_max:
                torch.save(model.state_dict(),'{}_{}_{}_maxacc'.format(model.name,activation_function,config['Optimizer']))
            
            if i % printstep == 0:
                print('train - epoch : {}, loss : {}, accurancy : {:.2f}'.format(i,train_loss,train_accuracy))
                print('test  - epoch : {}, loss : {}, accurancy : {:.2f}'.format(i,test_loss,test_accuracy))
        
        dfloss['{}_{}_train'.format(model.name,activation_function)] = train_loss_list
        dfloss['{}_{}_test'.format(model.name,activation_function)] = test_loss_list
        
        dfacc['{}_{}_train'.format(model.name,activation_function)] = train_accuracy_list
        dfacc['{}_{}_test'.format(model.name,activation_function)] = test_accuracy_list
        test_acc_max_list.append(test_acc_max)
        print('{}_{},best_train_acc : {}'.format(model.name,activation_function,train_acc_max))
        print('{}_{},best_test_acc : {}'.format(model.name,activation_function,test_acc_max))
        
    df_max.loc['{}'.format(model.name)] = test_acc_max_list
    
    plt.figure(figsize=(9,6))
    plt.plot(dfloss)
    plt.title('Loss Activation function comparision({})'.format(model.name), fontsize=12)
    plt.xlabel("Epoch",fontsize = 12)
    plt.ylabel("Loss",fontsize = 12)
    plt.legend(dfloss.columns.values)
    plt.savefig('{}_Loss.png'.format(model.name))
    

    plt.figure(figsize=(9,6))
    plt.plot(dfacc)
    plt.title('Accuracy Activation function comparision({})'.format(model.name), fontsize=12)
    plt.xlabel("Epoch",fontsize = 12)
    plt.ylabel("Accuracy(%)",fontsize = 12)
    plt.legend(dfacc.columns.values)
    plt.savefig('{}_Acc.png'.format(model.name))

print('Batch_size:{},optimizer:{},lr:{}'.format(config['Batch_size'],config['Optimizer'],config['Optim_hparas']))
print(df_max)
