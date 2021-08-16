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
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

def getdatafromtxt(path, mode):
    word = []
    with open('{}{}.txt'.format(path, mode), 'r') as file:
        for i in file:
            word.extend(i.split('\n')[0].split(' '))
    return word

def comptestlist(listt):
    t_t_l = [0,3,0,2,0,1,0,1,3,1,0,2,3,0,2,0,2,3,2,1]
    test = []
    for i in range(len(listt)):
        if i%2 == 0:
            test.append([[word2idx(listt[i]).to(device),torch.tensor([t_t_l[i]]).to(device)],
                         [word2idx(listt[i+1]).to(device),torch.tensor([t_t_l[i+1]]).to(device)]])
    return test

def tensorsFromPair(idx, train_list):
    t_n = idx%4
    ch_n = random.randint(1,3)
    ind_f = idx - t_n
    consin_c = ind_f + (t_n + ch_n)%4
    
    return [word2idx(train_list[idx]).to(device), torch.tensor([idx%4]).to(device)] , [word2idx(train_list[consin_c]).to(device), torch.tensor([consin_c%4]).to(device)]

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

char2idx_dict = creat_char2idx_dict()
idx2char_dict = creat_idx2char_dict()

def word2idx(word, eos = True):
    s = []
    for i in word:
        s.append(char2idx_dict[i])
    if eos:
        s.append(char2idx_dict['EOS'])
    return torch.tensor(s).view(-1,1) #行數量不知道所以設 -1

def idx2word(idx):
    word = ""
    
    for i in idx:
        if i.item() == 1: 
            break
        char = idx2char_dict[i.item()]
        word += char
    return word

def Reparameterization_Trick(self, mean, logvar):
    std = torch.exp(logvar/2)
    eps = torch.randn_like(std)
    return mean + eps * std

def teacher_force_ratio(epoch, total_epoch):
    return 1-epoch/total_epoch



MAX_LENGTH = 15
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, condition_size, latent_size):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.latent_size = latent_size
        
        
        self.embedding_init_c = nn.Embedding(4, condition_size)
#         self.init_h2encoder = nn.Linear(hidden_size + condition_size, hidden_size)
#         self.init_c2encoder = nn.Linear(hidden_size + condition_size, hidden_size)
        
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
        
        
    def forward(self, input_tensor, target_tensor, encoder_hidden, encoder_cell, teacher_forcing_ratio, criterion):
        
        input_length = input_tensor[0].size(0)
        target_length = target_tensor[0].size(0)
        CEloss = 0
        
        #----------sequence to sequence part for encoder----------#
        for en_idx in range(input_length):
            encoder_output, encoder_hidden, encoder_cell = self.encoder(input_tensor[0][en_idx], encoder_hidden, encoder_cell)
        
        #----------sequence to sequence part for latent----------#
        mean_h = self.hidden2mean(encoder_hidden)
        logvar_h = self.hidden2logvar(encoder_hidden)
        latent_h = self.Reparameterization_Trick(mean_h, logvar_h)
        decoder_hidden = self.latent2decoder_h(torch.cat((latent_h, self.embedding_init_c(target_tensor[1]).view(1, 1, -1)), dim = -1))
        KLloss_h = -0.5 * torch.sum(1 + logvar_h - mean_h**2 - logvar_h.exp())


        mean_c = self.cell2mean(encoder_cell)
        logvar_c = self.cell2logvar(encoder_cell)
        latent_c = self.Reparameterization_Trick(mean_c, logvar_c)
        decoder_cell = self.latent2decoder_c(torch.cat((latent_c, self.embedding_init_c(target_tensor[1]).view(1, 1, -1)), dim = -1))
        KLloss_c = -0.5 * torch.sum(1 + logvar_c - mean_c**2 - logvar_c.exp())

        KLloss = KLloss_h + KLloss_c
        
        
        decoder_input = torch.tensor([[SOS_token]], device=device)
        #----------sequence to sequence part for decoder----------#
#         predict_idx = []
#         pred_distribution = []
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
            for de_idx in range(target_length):
                decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
                CEloss += criterion(decoder_output, target_tensor[0][de_idx])
                decoder_input = target_tensor[0][de_idx]  # Teacher forcing
#                 predict_idx.append(decoder_output.tolist())

        else:
            # Without teacher forcing: use its own predictions as the next input
            for de_idx in range(target_length):
                decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                
                CEloss += criterion(decoder_output, target_tensor[0][de_idx])
                if decoder_input.item() == EOS_token:
                    break
        
        return CEloss/target_length, KLloss
    
    def eva8(self, input_tensor, target_tensor, encoder_hidden, encoder_cell):
        input_length = input_tensor[0].size(0)
        target_length = target_tensor[0].size(0)
        
        for en_idx in range(input_length):
            encoder_output, encoder_hidden, encoder_cell = self.encoder(input_tensor[0][en_idx], encoder_hidden, encoder_cell)
        
        mean_h = self.hidden2mean(encoder_hidden)
        logvar_h = self.hidden2logvar(encoder_hidden)
        latent_h = self.Reparameterization_Trick(mean_h, logvar_h)
        decoder_hidden = self.latent2decoder_h(torch.cat((latent_h, self.embedding_init_c(target_tensor[1]).view(1, 1, -1)), dim = -1))
        
        
        mean_c = self.cell2mean(encoder_cell)
        logvar_c = self.cell2logvar(encoder_cell)
        latent_c = self.Reparameterization_Trick(mean_c, logvar_c)
        decoder_cell = self.latent2decoder_c(torch.cat((latent_c, self.embedding_init_c(target_tensor[1]).view(1, 1, -1)), dim = -1))
        
        decoder_input = torch.tensor([[SOS_token]], device=device)
        
        pred_idx = []
        
        for de_idx in range(target_length):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            pred_idx.append(decoder_input.tolist())

            if decoder_input.item() == EOS_token:
                break
        return pred_idx
        
        
        
        
        
        
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
            return torch.zeros(1, 1, self.hidden_size - self.condition_size, device = device)

        def initCell(self):
            return torch.zeros(1, 1, self.hidden_size - self.condition_size, device = device)
        
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
        
def train(model, input_tensor, target_tensor, optimizer, criterion, teacher_force_ratio, kl_w, max_length=MAX_LENGTH):
    
    
    encoder_hidden = torch.cat((model.encoder.initHidden(), model.embedding_init_c(input_tensor[1]).view(1, 1, -1)), dim = -1)
    encoder_cell = torch.cat((model.encoder.initCell(), model.embedding_init_c(input_tensor[1]).view(1, 1, -1)), dim = -1)
    
    optimizer.zero_grad()
    CEloss, KLloss = model(input_tensor, target_tensor, encoder_hidden, encoder_cell, teacher_force_ratio, criterion)
    loss = CEloss + kl_w * KLloss
    loss.backward()
    optimizer.step()
    
    return CEloss, KLloss, loss

def test(model, testlist):
    
    bleu_Score = 0
    for test_choose in testlist:
        input_tensor = test_choose[0]
        target_tensor = test_choose[1]
#         print(input_tensor)
#         print(target_tensor)
        
        encoder_hidden = torch.cat((model.encoder.initHidden(), model.embedding_init_c(input_tensor[1]).view(1, 1, -1)), dim = -1)
        encoder_cell = torch.cat((model.encoder.initCell(), model.embedding_init_c(input_tensor[1]).view(1, 1, -1)), dim = -1)
        
        pred = model.eva8(input_tensor, target_tensor, encoder_hidden, encoder_cell)
        bleu_Score += compute_bleu(idx2word(torch.tensor(pred).to(device)), idx2word(target_tensor[0].to(device)))
        
        
        
        
    return bleu_Score/len(testlist)
    
    
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
    
    

#     encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    
    #----------sequence to sequence part for encoder----------#

    
#     decoder_hidden = encoder_hidden ??
#     decoder_cell = encoder_cell     ??


    #----------sequence to sequence part for decoder----------#
    


def trainIters(model, n_iters, LR, path, print_every=1000, plot_every=500):
    start = time.time()
    plot_celosses = []
    plot_kllosses = []
    plot_bleu = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    CEloss_t, KLloss_t = 0, 0

    optimizer = optim.SGD(model.parameters(), lr=LR)
    
    train_list = getdatafromtxt(path,'train')
    test_list = comptestlist(getdatafromtxt(path,'test'))
#     training_pairs = [tensorsFromPair(random.randint(0, len(train_list)), train_list) for i in range(n_iters)]
    
    criterion = nn.CrossEntropyLoss()
    

    for iter in tqdm(range(1, n_iters + 1)):
        training_pair = tensorsFromPair(random.randint(0, len(train_list)-1), train_list)
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        t_f_r = teacher_force_ratio(iter ,n_iters)
        
        model.train()
        CEloss, KLloss, loss = train(model, input_tensor, target_tensor, optimizer, criterion, 
                                         t_f_r, KLD_weight, max_length=MAX_LENGTH)
        
        model.eval() 
        bleu_score = test(model, test_list)
        
        
        CEloss_t += CEloss
        KLloss_t += KLloss
        print_loss_total += loss

        
        if iter % plot_every == 0:
            model.eval() 
            bleu_score = test(model, test_list)
            
            plot_celosses.append(CEloss_t/plot_every)
            plot_kllosses.append(KLloss_t/plot_every)
            plot_bleu.append(bleu_score)
            
            print('bleu : {}'.format(bleu_score))

        if iter % print_every == 0:
            
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))


    


    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
#----------Hyper Parameters----------#
hidden_size = 256
#The number of vocabulary
vocab_size = 28
condition_size = 8
latent_size = 32
empty_input_ratio = 0.05
KLD_weight = 0.0
LR = 0.1
path = ''

# train_list = getdatafromtxt('')
# training_pairs = [tensorsFromPair(random.randint(0, len(train_list)), train_list) for i in range(50)]

vae = VAE(vocab_size, hidden_size, condition_size, latent_size).to(device)
trainIters(vae, 75000, LR, path, print_every=1000)

