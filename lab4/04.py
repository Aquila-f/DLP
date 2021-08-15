

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
            encoder_output, hidden_new, cell_new = self.encoder(word_vac, hidden_new, cell_new)
        
        
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
            predict_idx.append(predict_idx_elem[0].tolist())

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
