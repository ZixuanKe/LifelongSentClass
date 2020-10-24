import torch
import sys
import torch
import numpy as np

import utils
import numpy as np

import torch
from public.data import *
import public.config
import random

# initialize seeds
torch.backends.cudnn.enabled = False
torch.manual_seed(config.args.seed)
np.random.seed(config.args.seed)
random.seed(config.args.seed)
if config.args.cuda:
    torch.cuda.manual_seed_all(config.args.seed)

class Net(torch.nn.Module):

    def __init__(self, inputsize,taskcla,weights_matrix,voc_size):
        super(Net, self).__init__()
        ncha,row,col=inputsize
        size = 32

        self.col_to_size=torch.nn.Linear(col,size)
        self.row_to_size=torch.nn.Linear(row,size)
        self.taskcla=taskcla


        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.drop1 = torch.nn.Dropout(0.2)
        self.padding = torch.nn.ReplicationPad2d(1)

        self.c1 = torch.nn.Conv2d(ncha, 64, kernel_size=2, stride=1, padding=0, bias=False)
        self.c2 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=1,  padding=0, bias=False)
        self.c3 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0, bias=False)

        self.fc1 = torch.nn.Linear(256 * 4 * 4, 1000, bias=False)
        self.fc2 = torch.nn.Linear(1000, 1000, bias=False)

        torch.nn.init.xavier_normal(self.fc1.weight)
        torch.nn.init.xavier_normal(self.fc2.weight)

        self.embedding = torch.nn.Embedding(voc_size, config.args.embedding_dim).cuda()
        weights_matrix = torch.FloatTensor(weights_matrix)
        self.embedding.from_pretrained(weights_matrix)
        self.embedding.weight.requires_grad = False # non trainable

        self.fc3=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.fc3.append(torch.nn.Linear(1000,n))

        return

    def forward(self, x,):
        h_list = []
        x_list = []
        # Gated
        x = self.embedding(x) # read word embedding
        x = torch.unsqueeze(x,1)

        h = self.col_to_size(x)
        h = h.transpose(2,3)
        h = self.row_to_size(h)
        h = h.view(-1,1,32,32)


        # x = self.padding(x)
        x_list.append(torch.mean(h, 0, True))
        con1 = self.drop1(self.relu(self.c1(h)))
        con1_p = self.maxpool(con1)

        con1_p = self.padding(con1_p)
        x_list.append(torch.mean(con1_p, 0, True))
        con2 = self.drop1(self.relu(self.c2(con1_p)))
        con2_p = self.maxpool(con2)

        con2_p = self.padding(con2_p)
        x_list.append(torch.mean(con2_p, 0, True))
        con3 = self.drop1(self.relu(self.c3(con2_p)))
        con3_p = self.maxpool(con3)

        h = con3_p.view(x.size(0), -1)
        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc1(h))

        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc2(h))

        h_list.append(torch.mean(h, 0, True))

        y=[]
        for i,_ in self.taskcla:
            y.append(self.fc3[i](h))

        return y, h_list, x_list
