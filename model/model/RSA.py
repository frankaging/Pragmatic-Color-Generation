import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

'''
Model class
'''
class LiteralSpeaker(nn.Module):
    def __init__(self,
                 input_dim=300,
                 color_dim=54,
                 output_dim=3,
                 hidden_1=30,
                 device=torch.device('cuda:0')):
        super(LiteralSpeaker, self).__init__()
        
        self.input_dim = input_dim
        self.color_dim = color_dim
        self.hidden_1 = hidden_1
        self.output_dim = output_dim

        self.color_fc1 = nn.Sequential(
                            nn.Linear(self.input_dim*2+self.color_dim, self.hidden_1),
                            nn.Sigmoid())

        # using matrix-wise ops
        self.channel1_m = nn.Sequential(
                       nn.Linear(self.hidden_1+self.color_dim, self.output_dim),
                       nn.Sigmoid())

        self.channel2_m = nn.Sequential(
                       nn.Linear(self.hidden_1+self.color_dim, self.output_dim),
                       nn.Sigmoid())

        self.channel3_m = nn.Sequential(
                       nn.Linear(self.hidden_1+self.color_dim, self.output_dim),
                       nn.Sigmoid())
        
        self.beta = nn.Sequential(
                            nn.Linear(self.hidden_1+self.color_dim, self.output_dim),
                            nn.Sigmoid())
        
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)
        
    def forward(self, emb1, emb2, source_color):
        x1 = self.color_fc1(torch.cat([emb1, emb2, source_color], dim=-1))
        channel1_m = self.channel1_m(torch.cat([x1, source_color], dim=-1))
        channel2_m = self.channel2_m(torch.cat([x1, source_color], dim=-1))
        channel3_m = self.channel3_m(torch.cat([x1, source_color], dim=-1))
        m = torch.stack([channel1_m, channel2_m, channel3_m], dim=1)
        r = source_color.unsqueeze(dim=-1)
        mr = torch.bmm(m, r).squeeze(dim=-1)
        beta = self.beta(torch.cat([x1, source_color], dim=-1))
        pred = mr + beta
        return pred

'''
Model class
'''
class LiteralListener(nn.Module):
    def __init__(self,
                 input_dim=300,
                 color_dim=54,
                 output_dim=3,
                 hidden_1=30,
                 device=torch.device('cuda:0')):
        super(LiteralListener, self).__init__()
        
        self.input_dim = input_dim
        self.color_dim = color_dim
        self.hidden_1 = hidden_1
        self.output_dim = output_dim

        self.color_fc1 = nn.Sequential(
                            nn.Linear(self.input_dim*2+self.color_dim, self.hidden_1),
                            nn.Sigmoid())

        # using matrix-wise ops
        self.channel1_m = nn.Sequential(
                       nn.Linear(self.hidden_1+self.color_dim, self.output_dim),
                       nn.Sigmoid())

        self.channel2_m = nn.Sequential(
                       nn.Linear(self.hidden_1+self.color_dim, self.output_dim),
                       nn.Sigmoid())

        self.channel3_m = nn.Sequential(
                       nn.Linear(self.hidden_1+self.color_dim, self.output_dim),
                       nn.Sigmoid())
        
        self.beta = nn.Sequential(
                            nn.Linear(self.hidden_1+self.color_dim, self.output_dim),
                            nn.Sigmoid())
        
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)
        
    def forward(self, emb1, emb2, source_color):
        x1 = self.color_fc1(torch.cat([emb1, emb2, source_color], dim=-1))
        channel1_m = self.channel1_m(torch.cat([x1, source_color], dim=-1))
        channel2_m = self.channel2_m(torch.cat([x1, source_color], dim=-1))
        channel3_m = self.channel3_m(torch.cat([x1, source_color], dim=-1))
        m = torch.stack([channel1_m, channel2_m, channel3_m], dim=1)
        r = source_color.unsqueeze(dim=-1)
        mr = torch.bmm(m, r).squeeze(dim=-1)
        beta = self.beta(torch.cat([x1, source_color], dim=-1))
        pred = mr + beta
        return pred