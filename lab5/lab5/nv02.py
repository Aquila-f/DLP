import json
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import matplotlib.animation as animation
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)

def get_iCLEVR_data(root_folder,mode):
    if mode == 'train':
        data = json.load(open(os.path.join(root_folder,'train.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        img = list(data.keys())
        label = list(data.values())
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return np.squeeze(img), np.squeeze(label)
    else:
        data = json.load(open(os.path.join(root_folder,'test.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        label = data
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return None, label


class ICLEVRLoader(data.Dataset):
    def __init__(self, root_folder, mode, trans=None, cond=False):
        self.root_folder = root_folder
        self.mode = mode
        self.img_list, self.label_list = get_iCLEVR_data(root_folder,mode)
        if self.mode == 'train':
            print("> Found %d images..." % (len(self.img_list)))
        
        self.cond = cond
        self.num_classes = 24
        
                
    def __len__(self):
        """'return the size of dataset"""
        return len(self.label_list)
        

    def __getitem__(self, index):
        train_preprocess = transforms.Compose([
            # transforms.CenterCrop(240),
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        if self.mode == 'train':
            path = 'images/{}'.format(self.img_list[index])
            img = Image.open(path).convert('RGB')
            re = train_preprocess(img)
            
        else:
            return self.label_list[index]
    
        return re, self.label_list[index]
    

def prep_dataloader(root, Batch_size):
    train_dataset = ICLEVRLoader(root, 'train')
    train_loader = data.DataLoader(
        train_dataset,
        batch_size = Batch_size,
        shuffle = True,
        num_workers = 7
    )
    
    test_dataset = ICLEVRLoader(root, 'test')
    test_loader = data.DataLoader(
        test_dataset,
        batch_size = Batch_size,
        shuffle = False,
        num_workers = 7
    )
    return train_loader, test_loader

def pr(loader):
    i = 0
    s = torch.tensor([])
    for a,b in loader:
        s = a if i == 0 else torch.cat((s, a),0)
        i += 1
        if i == 2: break
    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(s[-64:], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig('nv02-frist.png')
    


batch_size = 32
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 150
lrG = 0.001
lrD = 0.0003
beta1 = 0.5
ngpu = 7

train_loader, test_loader = prep_dataloader('', 32)


pr(train_loader)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# def randfake_label(labelloader):
#     z = torch.tensor([]).to(device)
#     for i in labelloader:
#         s = torch.randint(2 ,(1,24)).to(device, dtype = torch.float)
#         while torch.equal(s[0],i):
#             s = torch.randint(2 ,(1,24)).to(device, dtype = torch.float)
#         z = torch.cat((z, s),0)
#     return z
def randfake_label(labelloader):
    z = torch.tensor([]).to(device)
    for i in labelloader:
        s = torch.zeros(1,24).to(device, dtype = torch.float)
        for j in range(random.randint(1,3)):
            s[0][random.randint(0,23)] = 1 
        while torch.equal(i,s[0]):
            s = torch.zeros(1,24).to(device, dtype = torch.float)
            for j in range(random.randint(1,3)):
                s[0][random.randint(0,23)] = 1

        z = torch.cat((z, s),0)

    return z

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.firstconv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )
        self.main = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        out = self.firstconv(input)
        # print(out)
        # print(out.shape)
        out = self.main(out)
        # print(out.shape)
        return out

netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.conv = nn.Sequential(
#             torch.Size([32, 3, 64, 64])
            
            nn.Conv2d(3,32,4,2,1),
            nn.BatchNorm2d(32),
#             torch.Size([32, 32, 32, 32])
            
            nn.Conv2d(32,64,8,2,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
#             torch.Size([32, 64, 14, 14])
            
            nn.Conv2d(64,64,10,2,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
#             torch.Size([32, 64, 4, 4])
            
            nn.Flatten()
#             torch.Size([32, 1024])
        )
        self.lin = nn.Sequential(
            nn.Linear(1048, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, input, dlabel):
        out = self.conv(input)
        # print(dlabel.shape)
        # print(out.shape)
        out = torch.cat((out, dlabel), 1)
        # print(out.shape)
        out = self.lin(out)
        out = self.sig(out)
        return out

netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
for a in test_loader:
    fixed_noise = torch.randn(32, nz-24, 1, 1, device=device)
    fixed_noise = torch.cat((fixed_noise,a.view(32,24,1,1).to(device, dtype = torch.float)),1)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(train_loader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        ## Train with all-real batch

        condition = data[1].view(-1,24).to(device, dtype = torch.float)

        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        # print(real_cpu.shape)
        # print(condition.shape)
        output = netD(real_cpu, condition).view(-1)

        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()


        noise = torch.randn(b_size, nz-24, 1, 1, device=device)
        noise = torch.cat((noise, data[1].view(-1,24,1,1).to(device, dtype = torch.float)), 1)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach(), condition).view(-1)


        errD_fake = criterion(output, label)
        errD_fake.backward()        
        D_G_z1 = output.mean().item()


        
        output = netD(real_cpu, randfake_label(condition)).view(-1)
        errD_fake1 = criterion(output, label)
        errD_fake1.backward()
        D_G_z = output.mean().item()

        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake + errD_fake1
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake, condition).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i+1 % 500 == 0:
            print('[%3d/%d][%3d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1+D_G_z, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 3000 == 0) or ((epoch == num_epochs-1) and (i == len(train_loader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            plt.figure(figsize=(20,10))
            plt.axis("off")
            plt.imshow(np.transpose(vutils.make_grid(fake[:32], padding=2, normalize=True).cpu(),(1,2,0)))
            plt.savefig('nv02-{}.png'.format(epoch))

        iters += 1

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig('nv02-l.png')
