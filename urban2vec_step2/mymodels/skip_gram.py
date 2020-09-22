from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import torchvision.transforms.functional as TF

from tqdm import tqdm
import numpy as np
import json
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import skimage
import skimage.io
import skimage.transform
from PIL import Image
import time
import os
from os.path import join, exists
import copy
import random
from collections import OrderedDict
from sklearn.metrics import r2_score

from utils.image_dataset import BasicImageDataset, GroupedImageDataset

import torch.nn.functional as F
from torchvision.models import Inception3
from collections import namedtuple

_InceptionOuputs = namedtuple('InceptionOuputs', ['logits', 'aux_logits'])


"""
This model includes embedding of places, with a inception model to embed images into the same dimension.
"""

class PlaceSkipGram(nn.Module):
    def __init__(self, num_places=77744, num_words=756, embedding_dim=100):
        super(PlaceSkipGram, self).__init__()
        self.place_embedding = nn.Embedding(num_places,embedding_dim)
        self.word_embedding = nn.Embedding(num_words,embedding_dim)
    def forward(self, place_indices, word_indices):
        z1 = self.place_embedding(place_indices)
        #print(word_indices)
        z2 = self.word_embedding(word_indices)                    # N x embedding_dim
        prod = torch.sum(z1 * z2, dim=1)                         # N
        score = torch.sigmoid(prod)                              # N
        return score
class PlaceSkipGrammargin(nn.Module):
    def __init__(self, num_places=77744, num_words=756, embedding_dim=100):
        super(PlaceSkipGrammargin, self).__init__()
        self.place_embedding = nn.Embedding(num_places,embedding_dim)
        self.word_embedding = nn.Embedding(num_words,embedding_dim)
    def forward(self, place_indices, word_indices):
        z1 = self.place_embedding(place_indices)
        z2 = self.word_embedding(word_indices)                    # N x embedding_dim
        score=torch.nn.functional.pairwise_distance(z1,z2,p=2.0)               # N
        return score
    