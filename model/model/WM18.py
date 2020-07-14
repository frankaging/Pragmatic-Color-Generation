import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


'''
Model class for WM18 paper
'''
class WM18(nn.Module):
    def __init__(self,
                 input_dim=300,
                 color_dim=54,
                 output_dim=3,
                 hidden_1=30,
                 device=torch.device('cuda:0')):
        super(WM18, self).__init__()
        
        self.input_dim = input_dim
        self.color_dim = color_dim
        self.hidden_1 = hidden_1
        self.output_dim = output_dim

        self.color_fc1 = nn.Sequential(
                            nn.Linear(self.input_dim*2+self.color_dim, self.hidden_1),
                            nn.Sigmoid())
        
        self.color_fc2 = nn.Sequential(
                            nn.Linear(self.hidden_1+self.color_dim, self.output_dim),
                            nn.Sigmoid())

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)
        
    def forward(self, emb1, emb2, source_color):
        x1 = self.color_fc1(torch.cat([emb1, emb2, source_color], dim=-1))
        pred = self.color_fc2(torch.cat([x1, source_color], dim=-1))
        return pred