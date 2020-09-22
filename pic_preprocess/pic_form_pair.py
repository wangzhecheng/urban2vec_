# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:22:50 2019

@author: lhy
"""

import pickle
import random
import copy
import csv
import os
import re
with open("context_knn_chi.pickle","rb") as file:
    context=pickle.load(file)
with open("data_knn_chi.pickle","rb") as file:
    data=pickle.load(file)
context_n=copy.deepcopy(context)
path='C:/lhy/urban2vec/meta_chicago/meta_data/image_meta'
for i in context.keys():
    for j in context[i]:
        if j in context_n.keys() and i not in context_n[j]:
            context_n[j].append(i)
        if j not in context_n.keys():
            context_n[j]=[i]
            media=re.split('[/.]',j)
            with open(os.path.join(path,str(media[0])+'.csv'),'r') as file:
                csvr=csv.reader(file)
                count=0
                for row in csvr:
                    count=count+1
                    if count==int(media[1])+2:
                        data.append([j,float(row[5]),float(row[6]),int(row[2])])
                        break
context=context_n
random.shuffle(data)
train_set=[]
for i in range(50000):
    train_set.append(data[i][0])
val_set=[]
for i in range(50000,64739):
    val_set.append(data[i][0])
train_pair=[]
train_count=0
for i in range(len(data)):
    if data[i][0] in train_set:
        for j in range(len(context[data[i][0]])):
            if context[data[i][0]][j] in train_set:
                train_count=train_count+1
                train_pair.append([data[i][0],context[data[i][0]][j],data[i][3]])
val_pair=[]
val_count=0
for i in range(len(data)):
    if data[i][0] in val_set:
        for j in range(len(context[data[i][0]])):
            if context[data[i][0]][j] in val_set:
                val_count=val_count+1
                val_pair.append([data[i][0],context[data[i][0]][j],data[i][3]])

train_pair=random.sample(train_pair,100000)
val_pair=random.sample(val_pair,20000)
"""
with open("train_pair_knn_ny.pickle","wb") as file:
    pickle.dump(train_pair,file)
with open("val_pair_knn_ny.pickle","wb") as file:
    pickle.dump(val_pair,file)
"""