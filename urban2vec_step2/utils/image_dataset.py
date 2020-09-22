from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import torchvision.transforms.functional as TF

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
import copy

class BasicImageDataset(Dataset):
    default_output_cols = ['population_density', 'average_household_income', 'employ_rate', 'gini_index',
                           'poverty_family_below_poverty_level_rate', 'housing_unit_median_value',
                           'housing_unit_median_gross_rent', 'age_median', 'voting_2016_dem_percentage',
                           'voting_2016_gop_percentage',
                           'number_of_years_of_education', 'diversity', 'occupancy_vacant_rate', 'occupancy_owner_rate',
                           'mortgage_with_rate',
                           'dropout_16_19_inschool_rate', 'health_insurance_none_rate',
                           'education_less_than_high_school_rate',
                           'race_white_rate', 'race_black_africa_rate', 'race_indian_alaska_rate', 'race_asian_rate',
                           'race_islander_rate',
                           'race_other_rate', 'race_two_more_rate', 'average_household_size',
                           'household_type_family_rate']

    def __init__(self, root_dir, path_list, response_df, transform, response_cols=None):
        self.root_dir = root_dir
        self.path_list = path_list
        self.response_df = copy.deepcopy(response_df)
        self.transform = transform
        self.response_df.index = self.response_df['census_tract_fips']
        self.response_cols = response_cols
        if self.response_cols == None:
            self.response_cols = BasicImageDataset.default_output_cols

        # normalize the response variables
        self.response_mean = np.zeros(len(self.response_cols))
        self.response_std = np.zeros(len(self.response_cols))
        for i, col in enumerate(self.response_cols):
            self.response_mean[i] = self.response_df[col].mean()
            self.response_std[i] = self.response_df[col].std()
            self.response_df[col] = (self.response_df[col] - self.response_df[col].mean()) * 1. / self.response_df[
                col].std()

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        fips, subpath = self.path_list[idx]
        img_path = os.path.join(self.root_dir, subpath)
        response = self.response_df.loc[int(fips), self.response_cols].values
        image = Image.open(img_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        sample = [image,
                  torch.tensor(response, dtype=torch.float)]
        return sample


class GroupedImageDataset(Dataset):
    default_output_cols = ['population_density', 'average_household_income', 'employ_rate', 'gini_index',
                           'poverty_family_below_poverty_level_rate', 'housing_unit_median_value',
                           'housing_unit_median_gross_rent', 'age_median', 'voting_2016_dem_percentage',
                           'voting_2016_gop_percentage',
                           'number_of_years_of_education', 'diversity', 'occupancy_vacant_rate', 'occupancy_owner_rate',
                           'mortgage_with_rate',
                           'dropout_16_19_inschool_rate', 'health_insurance_none_rate',
                           'education_less_than_high_school_rate',
                           'race_white_rate', 'race_black_africa_rate', 'race_indian_alaska_rate', 'race_asian_rate',
                           'race_islander_rate',
                           'race_other_rate', 'race_two_more_rate', 'average_household_size',
                           'household_type_family_rate']

    def __init__(self, root_dir, path_list, response_df, transform, response_cols=None):
        self.root_dir = root_dir
        self.path_list = path_list
        self.response_df = copy.deepcopy(response_df)
        self.transform = transform
        self.response_df.index = self.response_df['census_tract_fips']
        self.response_cols = response_cols
        if self.response_cols == None:
            self.response_cols = GroupedImageDataset.default_output_cols

        # normalize the response variables
        self.response_mean = np.zeros(len(self.response_cols))
        self.response_std = np.zeros(len(self.response_cols))
        for i, col in enumerate(self.response_cols):
            self.response_mean[i] = self.response_df[col].mean()
            self.response_std[i] = self.response_df[col].std()
            self.response_df[col] = (self.response_df[col] - self.response_df[col].mean()) * 1. / self.response_df[
                col].std()

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        fips, subpath_list = self.path_list[idx]
        image_list = []
        for subpath in subpath_list:
            img_path = os.path.join(self.root_dir, subpath)
            image = Image.open(img_path)
            if not image.mode == 'RGB':
                image = image.convert('RGB')
            image = self.transform(image).unsqueeze(0)
            image_list.append(image)
        image_group = torch.cat(image_list, dim=0)
        response = self.response_df.loc[int(fips), self.response_cols].values
        sample = [image_group,
                  torch.tensor(response, dtype=torch.float)]
        return sample


class PlacePairDataset(Dataset):
    def __init__(self, path_list,weight,id_list,num):
        self.path_list = path_list
        self.weight=weight
        self.id_list=id_list
        self.num=num
    def __len__(self):
        return len(self.path_list)
    def __getitem__(self, idx):
        pos_idx,place_idx = self.path_list[idx]
        media1=copy.deepcopy(self.weight)
        media2=np.zeros(self.num)
        media2[self.id_list[place_idx]]=self.weight[self.id_list[place_idx]]
        media1=media1-media2
        media1=np.power(media1,0.5)
        media1=media1/np.sum(media1)
        neg_idx=np.random.choice(np.arange(self.num),size=1,p=media1)[0]
        sample = [torch.tensor(place_idx, dtype=torch.long), torch.tensor(pos_idx, dtype=torch.long), torch.tensor(neg_idx, dtype=torch.long)]
        return sample
