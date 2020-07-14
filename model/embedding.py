#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (4.0, 0.5) 
import matplotlib.patches as mpatches
from util.color_util import *
import pickle


# In[2]:


triple_train = pickle.load( open( "../munroe/triple_train.p", "rb" ) )
triple_dev = pickle.load( open( "../munroe/triple_dev.p", "rb" ) )
triple_test = pickle.load( open( "../munroe/triple_test.p", "rb" ) )


# In[3]:


glove42B = glove2dict("../glove.42B/glove.42B.300d.txt")


# In[4]:


# use this lib to make some corner case work.
modifier_map = {
    "vivider" : "vivid"
}


# In[9]:


# triple_test[:5]


# In[7]:


glove_color = dict()
# lazy load data
for tri in triple_train:
    modifier = tri[1]
    base = tri[2]
    for w in modifier:
        if w not in glove42B.keys():
            glove_color[w] = glove42B[modifier_map[w]]
        else:
            glove_color[w] = glove42B[w]
    for w in base:
        if w not in glove42B.keys():
            glove_color[w] = glove42B[modifier_map[w]]
        else:
            glove_color[w] = glove42B[w]
for tri in triple_dev:
    modifier = tri[1]
    base = tri[2]
    for w in modifier:
        if w not in glove42B.keys():
            glove_color[w] = glove42B[modifier_map[w]]
        else:
            glove_color[w] = glove42B[w]
    for w in base:
        if w not in glove42B.keys():
            glove_color[w] = glove42B[modifier_map[w]]
        else:
            glove_color[w] = glove42B[w]
for tri in triple_test:
    modifier = tri[1]
    base = tri[2]
    for w in modifier:
        if w not in glove42B.keys():
            glove_color[w] = glove42B[modifier_map[w]]
        else:
            glove_color[w] = glove42B[w]
    for w in base:
        if w not in glove42B.keys():
            glove_color[w] = glove42B[modifier_map[w]]
        else:
            glove_color[w] = glove42B[w]


# In[8]:


# print(len(glove_color))


# In[10]:


pickle.dump( glove_color, open( "../munroe/glove_color.p", "wb" ) )

