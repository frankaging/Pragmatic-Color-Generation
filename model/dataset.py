#!/usr/bin/env python
# coding: utf-8

from util.color_util import *

# In[70]:


'''
Generate the dataset needed for the model
TODO: Put it in as a inherent of the base pytorch dataset class
'''
import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (4.0, 0.5) 
import matplotlib.patches as mpatches


# In[2]:


BASE_DIR = "../munroe"


# ## Load labels, words and color mapping into memory

# In[78]:


# load label to word dictionary
label2words = {}
file_w2l = os.path.join(BASE_DIR, "words_to_labels.txt")
with open(file_w2l,encoding="utf-8") as f:
    for line in f:
        words, label = line.strip().split(',')
        label2words[label] = words


# In[36]:


train, dev, test = load_splits()
# load color map
cdict_train = load_rgb(train)
cdict_dev = load_rgb(dev)
cdict_test = load_rgb(test)


# In[79]:


triple_train = load_triple(cdict_train, label2words)
triple_dev = load_triple(cdict_dev, label2words)
triple_test = load_triple(cdict_test, label2words)


# ## Saving all findings into disk for training use

# In[ ]:

import pickle
pickle.dump( triple_train, open( "../munroe/triple_train.p", "wb" ) )
pickle.dump( triple_dev, open( "../munroe/triple_dev.p", "wb" ) )
pickle.dump( triple_test, open( "../munroe/triple_test.p", "wb" ) )

pickle.dump( cdict_train, open( "../munroe/cdict_train.p", "wb" ) )
pickle.dump( cdict_dev, open( "../munroe/cdict_dev.p", "wb" ) )
pickle.dump( cdict_test, open( "../munroe/cdict_test.p", "wb" ) )

# non-extend version
triple_train_shrink = load_triple(cdict_train, label2words, extend=False)
triple_dev_shrink = load_triple(cdict_dev, label2words, extend=False)
triple_test_shrink = load_triple(cdict_test, label2words, extend=False)
pickle.dump( triple_train_shrink, open( "../munroe/triple_train_reduce.p", "wb" ) )
pickle.dump( triple_dev_shrink, open( "../munroe/triple_dev_reduce.p", "wb" ) )
pickle.dump( triple_test_shrink, open( "../munroe/triple_test_reduce.p", "wb" ) )

