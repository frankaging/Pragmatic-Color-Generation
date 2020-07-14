import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

'''
Model classes for HSC19 paper
'''
class HSC19_RGB(nn.Module):
    def __init__(self,
                 input_dim=300,
                 color_dim=3,
                 output_dim=3,
                 hidden_1=30,
                 hidden_2=10,
                 device=torch.device('cuda:0')):
        super(HSC19_RGB, self).__init__()
        
        self.input_dim = input_dim
        self.color_dim = color_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.output_dim = output_dim

        self.color_fc1 = nn.Sequential(
                            nn.Linear(self.input_dim*2+self.color_dim, self.hidden_1),
                            nn.Sigmoid())
        
        self.color_fc2 = nn.Sequential(
                            nn.Linear(self.hidden_1, self.hidden_2),
                            nn.Sigmoid())

        self.m1 = nn.Sequential(
                    nn.Linear(self.hidden_2, self.output_dim),
                    nn.Sigmoid())

        self.m2 = nn.Sequential(
                    nn.Linear(self.hidden_2, self.output_dim),
                    nn.Sigmoid())

        self.m3 = nn.Sequential(
                    nn.Linear(self.hidden_2, self.output_dim),
                    nn.Sigmoid())

        self.beta = nn.Sequential(
                    nn.Linear(self.hidden_2, self.output_dim),
                    nn.Sigmoid())
        
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)
        
    def forward(self, emb1, emb2, source_color):

        # feed forward layers
        x1 = self.color_fc1(torch.cat([emb1, emb2, source_color], dim=-1))
        x2 = self.color_fc2(x1)
        # m matrix
        m1 = self.m1(x2)
        m2 = self.m2(x2)
        m3 = self.m3(x2)
        m = torch.stack([m1,m2,m3], dim=1)
        # mr
        r = source_color.unsqueeze(dim=-1)
        mr = torch.bmm(m, r).squeeze(dim=-1)
        beta = self.beta(x2)
        pred = mr + beta

        return pred

'''
Model classes for HSC19 paper
'''
class HSC19_HSV(nn.Module):
    def __init__(self,
                 input_dim=300,
                 color_dim=3,
                 output_dim=3,
                 hidden_1=32,
                 hidden_2=16,
                 device=torch.device('cuda:0')):
        super(HSC19_HSV, self).__init__()
        
        self.input_dim = input_dim
        self.color_dim = color_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.output_dim = output_dim

        self.color_fc1 = nn.Sequential(
                            nn.Linear(self.input_dim*2+self.color_dim, self.hidden_1),
                            nn.Sigmoid())
        
        self.color_fc2 = nn.Sequential(
                            nn.Linear(self.hidden_1, self.hidden_2),
                            nn.Sigmoid())

        self.m1 = nn.Sequential(
                    nn.Linear(self.hidden_2, self.output_dim),
                    nn.Sigmoid())

        self.m2 = nn.Sequential(
                    nn.Linear(self.hidden_2, self.output_dim),
                    nn.Sigmoid())

        self.m3 = nn.Sequential(
                    nn.Linear(self.hidden_2, self.output_dim),
                    nn.Sigmoid())

        self.h = nn.Sequential(
                    nn.Linear(1, 1),
                    nn.Sigmoid())
        
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)
        
    def forward(self, emb1, emb2, source_color):

        # feed forward layers
        x1 = self.color_fc1(torch.cat([emb1, emb2, source_color], dim=-1))
        x2 = self.color_fc2(x1)
        # m matrix
        m1 = self.m1(x2)
        m2 = self.m2(x2)
        m3 = self.m3(x2)
        m = torch.stack([m1,m2,m3], dim=1)
        # mr
        r = source_color.unsqueeze(dim=-1)
        mr = torch.bmm(m, r)
        # h
        h = self.h(mr[:,0])
        sv = mr[:,1:].squeeze(dim=-1)

        return h, sv