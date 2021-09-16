# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 10:23:13 2021

@author: Akshay Gupta
"""

import torch
import torch.nn as nn


import warnings
from api.processor.inception import InceptionBlock

warnings.filterwarnings("ignore")
# os.chdir(path)

#%%


class Flatten(nn.Module):
    def __init__(self, out_features):
        super(Flatten, self).__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)


class Reshape(nn.Module):
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return x.view(-1, *self.out_shape)


LOOK_BACK = 15
model = nn.Sequential(Reshape(out_shape=(5, LOOK_BACK)), InceptionBlock(in_channels=5, n_filters=32, kernel_sizes=[5,13,23],
                                                                        bottleneck_channels=32, use_residual=True, activation=nn.ReLU()),
                      InceptionBlock(in_channels=32*4, n_filters=32, kernel_sizes=[5, 13, 23], bottleneck_channels=32, use_residual=True,
                                     activation=nn.ReLU()), nn.AdaptiveMaxPool1d(output_size=1), Flatten(out_features=32*4*1),
                      nn.Linear(in_features=4*32*1, out_features=3))

model.load_state_dict(torch.load('Inception_time.pth', map_location=torch.device('cpu')))
model.eval()
