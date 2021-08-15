from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import torch.utils.data as data
print(torch.__version__)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def txt2list(mode):
    status = True if mode == 'train' else False
    word = []
    tense = []    
    with open('{}.txt'.format(mode), 'r') as file:
        if status:
            for i in file:
                word.extend(i.split('\n')[0].split(' '))
                tense.extend([0,1,2,3])
        else:
            for idx, i in enumerate(file):
                word.append(i.split('\n')[0].split(' '))
            tense = [[0,3],[0,2],[0,1],[0,1],[3,1],[0,2],[3,0],[2,0],[2,3],[2,1]]
    return word, tense

def creat_char2idx_dict():
    s = {'SOS':0,'EOS':1}
    for i in range(26):
        s.setdefault(chr(i+97),i+2)
    return s

def creat_idx2char_dict():
    s = {0:'SOS',1:'EOS'}
    for i in range(26):
        s.setdefault(i+2,chr(i+97))
    return s

def word2idx(word, eos = True):
    s = []
    for i in word:
        s.append(char2idx_dict[i])
    if eos:
        s.append(char2idx_dict['EOS'])
    return torch.tensor(s).view(-1,1) #行數量不知道所以設 -1

def tense2idx(tense):
    return torch.tensor([tense])

def idx2word(idx):
    word = ""
    for i in idx:
        if i.item() == 1: break
        char = idx2char_dict[i.item()]
        word += char
    return word

char2idx_dict = creat_char2idx_dict()
idx2char_dict = creat_idx2char_dict()

def get_max_len(word):
    max_len = 0
    for w in word:
        max_len = max(max_len, len(w))
    return max_len

class Datasetloader(data.Dataset):
    def __init__(self, path, mode):
        self.word, self.tense = txt2list(mode)
        self.char2idx_dict = creat_char2idx_dict()
        self.idx2char_dict = creat_idx2char_dict()
        self.max_len = get_max_len(self.word)
        self.train = True if mode == 'train' else False
        print('> {} find {} words...'.format(mode,len(self.word)))

    def __len__(self):
        return len(self.word)
    
    def __getitem__(self, index):        
        
        if self.train:
            return word2idx(self.word[index]),self.tense[index]
        return word2idx(self.word[index][0]), word2idx(self.word[index][1]), self.tense[index][0], self.tense[index][1]
    
def prep_dataloader(path):
    train_dataset = Datasetloader(path, 'train')
    train_loader = data.DataLoader(
        train_dataset,
        batch_size = 1,
        shuffle = True,
        num_workers = 4
    )
    
    test_dataset = Datasetloader(path, 'test')
    test_loader = data.DataLoader(
        test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 4
    )
    return train_loader, test_loader, train_dataset.max_len

def teacher_force_ratio(epoch, total_epoch):
    return 1-epoch/total_epoch

def kl_weight(epoch, total_epoch, kl_type, time):
    
    if kl_annealing_type == 'monotonic':
        return (1./(time-1))*(epoch-1) if epoch<time else 1.

    else: #cycle
        period = epochs//time
        epoch %= period
        KL_weight = sigmoid((epoch - period // 2) / (period // 10)) / 2
        return KL_weight
    
def got_ce_kl_loss(mean_h, logvar_h, mean_c, logvar_c, pred_output, target):
    loss_fun = nn.CrossEntropyLoss()
    CEloss = loss_fun(pred_output[:len(target)], target[:len(target)])
    
    KLloss_h = -0.5 * torch.sum(1 + logvar_h - mean_h**2 - logvar_h.exp())
    KLloss_c = -0.5 * torch.sum(1 + logvar_c - mean_c**2 - logvar_c.exp())
    
    return CEloss, KLloss_h + KLloss_c

def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, condition_size, latent_size):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.latent_size = latent_size
        
        
        self.embedding_init_c = nn.Embedding(4, condition_size)
        self.init_h2encoder = nn.Linear(hidden_size + condition_size, hidden_size)
        self.init_c2encoder = nn.Linear(hidden_size + condition_size, hidden_size)
        
        self.encoder = self.EncoderRNN(input_size, hidden_size, condition_size)
        self.decoder = self.DecoderRNN(input_size, hidden_size, condition_size)
        self.hidden2mean = nn.Linear(hidden_size, latent_size)
        self.hidden2logvar = nn.Linear(hidden_size, latent_size)
        self.cell2mean = nn.Linear(hidden_size, latent_size)
        self.cell2logvar = nn.Linear(hidden_size, latent_size)
        self.latent2decoder_h = nn.Linear(latent_size + condition_size, hidden_size)
        self.latent2decoder_c = nn.Linear(latent_size + condition_size, hidden_size)
        
        
    def Reparameterization_Trick(self, mean, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mean + eps * std
        
        
    def forward(self, word, tense, hidden, cell, teacher_forcing_ratio):
        
        emb_tense_init = self.embedding_init_c(tense).view(1, 1, -1)
        hidden_new = self.init_h2encoder(torch.cat((hidden, emb_tense_init), -1))
        cell_new = self.init_c2encoder(torch.cat((cell, emb_tense_init), -1))
        
        
        for word_vac in word[0]:
            encoder_output, encoder_hidden, encoder_cell = self.encoder(word_vac, hidden_new, cell_new)
        
        
        mean_h = self.hidden2mean(encoder_hidden)
        logvar_h = self.hidden2logvar(encoder_hidden)
        latent_h = self.Reparameterization_Trick(mean_h, logvar_h)
        decoder_h_init = self.latent2decoder_h(torch.cat((latent_h, emb_tense_init), -1))
        
        
        mean_c = self.cell2mean(encoder_cell)
        logvar_c = self.cell2logvar(encoder_cell)
        latent_c = self.Reparameterization_Trick(mean_c, logvar_c)
        decoder_c_init = self.latent2decoder_h(torch.cat((latent_c, emb_tense_init), -1))
        
        decoder_input = torch.tensor([[SOS_token]], device=device)
        predict_idx = []
        pred_distribution = []
        
        teacher_force = True if random.random() < teacher_forcing_ratio else False
        
        
        for id_d in range(len(word[0])):
            output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_h_init,decoder_c_init)
            pred_distribution.append(output[0].tolist())
            predict_idx_elem = output.topk(1)[1]
            predict_idx.append(predict_idx_elem)

            if teacher_force:
                decoder_input = word[0][id_d]
            else:
                if word[0][id_d].item() == EOS_token:
                    break
                decoder_input = predict_idx_elem
     
        return predict_idx, pred_distribution, mean_h, logvar_h, mean_c, logvar_c
        
        
        
    class EncoderRNN(nn.Module):
        def __init__(self, input_size, hidden_size, condition_size):
            super(VAE.EncoderRNN, self).__init__()
            
            self.hidden_size = hidden_size
            self.condition_size = condition_size
            
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size)

        def forward(self, input, hidden, cell):
            embed = self.embedding(input).view(1, 1, -1)
            output, (hidden, cell) = self.lstm(embed, (hidden, cell))

            return output, hidden, cell

        def initHidden(self):
            return torch.zeros(1, 1, self.hidden_size, device = device)

        def initCell(self):
            return torch.zeros(1, 1, self.hidden_size, device = device)
        
    class DecoderRNN(nn.Module):
        def __init__(self, input_size, hidden_size, condition_size):
            super(VAE.DecoderRNN, self).__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size)
            self.out = nn.Linear(hidden_size, input_size)
            self.softmax = nn.LogSoftmax(dim=1)

        def forward(self, input, hidden, cell):
            output = self.embedding(input).view(1, 1, -1)
            output = F.relu(output)
            output, (decoder_hidden, decoder_cell) = self.lstm(output, (hidden, cell))
            output = self.out(output[0])
            output = self.softmax(output)
            return output, decoder_hidden, decoder_cell

        def initHidden(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)

def train(model, train_loader, teacher_force_ratio, kl_weight, device):
    init_hidden = model.EncoderRNN.initHidden(model)
    init_cell = model.EncoderRNN.initCell(model)
    total_CEloss, total_KLloss, total_bluescore = 0, 0, 0
    
    for word, tense in tqdm(train_loader):
        optimizer.zero_grad()
        word, tense = word.to(device), tense.to(device)
        predict_idx, pred_one_hot, mean_h, logvar_h, mean_c, logvar_c = model(word, tense, init_hidden, init_cell, teacher_force_ratio)
        CEloss, KLloss = got_ce_kl_loss(mean_h, logvar_h, mean_c, logvar_c, torch.tensor(pred_one_hot), word.view(-1))
        total_CEloss += CEloss.item()
        total_KLloss += KLloss.item()
        loss = CEloss + kl_weight * KLloss
        loss.backward()
        optimizer.step()

        pred = idx2word(torch.tensor(predict_idx))
        label = idx2word(word[0])

        total_bluescore += compute_bleu(pred, label)

        
    return total_CEloss/len(train_loader), total_KLloss/len(train_loader), total_bluescore/len(train_loader)

# def test():
    
    

################################################

SOS_token = 0
EOS_token = 1

input_size = 28
hidden_size = 256
condition_size = 8
latent_size = 32
total_epochs = 50
print_step = 50

train_loader, test_loader, train_max_len = prep_dataloader('')
learn_rate = 0.05
# kl_annealing_type = 'cycle'

################################################

CEloss_list = []
KLloss_list = []
bluescore_list = []
teacher_list = []
kl_weight_list = []


ernn = VAE(input_size, hidden_size, condition_size, latent_size)
ernn.cuda()
optimizer = optim.SGD(ernn.parameters(), lr = learn_rate)

for epoch in range(total_epochs):
    
    ernn.train()
    teacher_force = teacher_force_ratio(epoch, total_epochs)
    kl_weight = 1
    CEloss, KLloss, blue_score= train(ernn, train_loader, teacher_force, kl_weight, device)    
    CEloss_list.append(CEloss)
    KLloss_list.append(KLloss)
    bluescore_list.append(blue_score)
    teacher_list.append(teacher_force)
    kl_weight_list.append(kl_weight)
    print(CEloss, KLloss, blue_score)
    







