# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:26:00 2019

@author: lhy
"""

import pickle
import numpy as np
from sklearn.decomposition import PCA
import csv
import numpy as np
from sklearn.preprocessing import scale
import statsmodels.api as sm
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
from matplotlib import gridspec
with open('tonumbar.pickle','rb') as file:
    tonum=pickle.load(file)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint=torch.load("sv_embedding_10_bar.tar",map_location=device)
place_embedding=np.array(checkpoint['model_state_dict']['place_embedding.weight'])
word_embedding=np.array(checkpoint['model_state_dict']['word_embedding.weight'])
pca=PCA(n_components=50)
pca.fit(place_embedding)
place_embedding=pca.transform(place_embedding)
"""
#回归
census=[]
census_index=[]
keyss=[]
variable_list=[2,6,11,27]
with open("census_ba_sub.csv") as file:
    countt=0
    f_csv=csv.reader(file)
    for row in f_csv:
        if countt==0:
            for k in range(1,28):
                if k in variable_list:
                    keyss.append(row[k])
            countt=countt+1
        else:
            if int(row[28]) in tonum['id'].keys():
                media=[]
                for k in range(1,28):
                    if k in variable_list:
                        if row[k] == "" :
                            media.append(None)
                        else:
                            media.append(float(row[k]))
                census_index.append(int(row[28]))    
                census.append(media)
for i in range(len(keyss)):
    summ=0.0
    count=0.0
    for j in range(len(census)):
        if census[j][i]:
            count=count+1.0
            summ=summ+census[j][i]
    mean=summ/count
    for j in range(len(census)):
        if census[j][i] is None:
            census[j][i]=mean
census=scale(np.array(census))
x=[None]*1197
for i in range(len(census_index)):
    x[i]=place_embedding[tonum['id'][census_index[i]]]
x=np.array(x)
x=scale(x)
for i in range(len(keyss)):
    x1=sm.add_constant(x)
    model = sm.OLS(census[:,i], x1)
    results = model.fit()
    print(keyss[i])
    print(results.summary())
"""
#展示词
word_embedding=word_embedding[0:767]
word_embedding=pca.transform(word_embedding)
low=[]
high=[]
rank=np.argsort(word_embedding[:,0])
word_index=[None]*word_embedding.shape[0]
for i in tonum['word'].keys():
    if tonum['word'][i]<len(word_index):
        word_index[tonum['word'][i]]=i
for i in range(7):
    low.append(word_index[rank[i]])
for i in range(7):
    high.append(word_index[rank[len(rank)-1-i]])
"""
#展示图
with open("embedding_ba.pickle","rb") as file:
    pic_embedding=pickle.load(file)
with open("image_ba.pickle","rb") as file:
    fips1=pickle.load(file)
pic_embedding=pca.transform(pic_embedding)
rank=np.argsort(pic_embedding[:,2])
low=[]
high=[]
for i in range(4):
    low.append(fips1[rank[i]][0])
for i in range(4):
    high.append(fips1[rank[len(rank)-1-i]][0])
image_path='C:/lhy/urban2vec/street_view/bay_area'
count=0
nrow = 2
ncol = 4
plt.rcParams['savefig.dpi'] = 200
fig = plt.figure(figsize=(2*ncol+1, 2*nrow+1)) 
gs = gridspec.GridSpec(nrow, ncol,
         wspace=0.03, hspace=0.10, 
         top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
         left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 
for i in low:
    im=Image.open(os.path.join(image_path,i))
    ax=plt.subplot(gs[0,count])
    count+=1
    ax.imshow(im)
    ax.axis('off')
count=0
for i in high:
    im=Image.open(os.path.join(image_path,i))
    #print("high"+i)
    ax=plt.subplot(gs[1,count])
    count+=1
    ax.imshow(im)
    ax.axis('off')
#plt.subplots_adjust(wspace=0.05,hspace=0.0)
plt.show()
"""