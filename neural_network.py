#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import random
############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
random.seed(123)
torch.manual_seed(123)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(16, 64)      
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)       

    def forward(self, input):
        input = F.relu(self.fc1(input))           
        input = F.relu(self.fc2(input))
        input = self.fc3(input)                 
        return input

net = Network()

############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = optim.Adam(net.parameters(), lr=0.0005)

loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

net.apply(weights_init)
scheduler = None

############################################################################
#######              Metaparameters and training options              ######
############################################################################

scale_inputs = True
train_val_split = 0.8
batch_size = 64
epochs = 100
