# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:09:23 2019

@author: lhy
"""

import numpy as np
import torch
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
from IPython.core.pylabtools import figsize
figsize(13.0,4.0)
plt.rcParams['savefig.dpi'] = 200
plt.subplots_adjust(wspace=0.03)
with open("tonumchir.pickle","rb") as file:
    id2num=pickle.load(file)["id"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint=torch.load("sv_embedding_10_chir.tar",map_location=device)
embedding=np.array(checkpoint['model_state_dict']['place_embedding.weight'])
cluster_x=embedding
kmeans=KMeans(n_clusters=4).fit(cluster_x)
col_list=["r","g","b","orange","purple","yellow","black","brown"]
with open("census_tract_business_dict_chi.pickle","rb") as file:
    data=pickle.load(file)
data_n={}
for i in data.keys():
    for j in data[i].keys():
        data[i][j]['fips']=int(i)
        data_n[j]=data[i][j]
del(data)
fips_unique=[]
fips_dict={}
for i in data_n.keys():
    if data_n[i]['fips'] not in fips_unique:
        fips_unique.append(data_n[i]['fips'])
        fips_dict[data_n[i]['fips']]=[i]
    else:
        fips_dict[data_n[i]['fips']].append(i)
dest=[None]*len(id2num)
cluster_x=[]
result=[None]*len(id2num)
label=[None]*len(id2num)
for i in id2num.keys():
    if i in fips_unique:
        x=[]
        destin=[[],[]]
        for j in fips_dict[i]:
            destin[0].append(data_n[j]['coordinates']['latitude'])
            destin[1].append(data_n[j]['coordinates']['longitude'])
        destin=np.array(destin).T
        dest[id2num[i]]=np.mean(destin,axis=0)
        result[id2num[i]]=kmeans.labels_[id2num[i]]
        a=True
        if a:
            label[id2num[i]]=0
        else:
            label[id2num[i]]=1
result=np.array(result)
label=np.array(label)
dest=np.array(dest)
c_list=[]
for i in range(result.shape[0]):
    c_list.append(col_list[result[i]])
for j in range(1):
    dest_media=dest[np.where(label==j)]
    result_media=result[np.where(label==j)]
    plt.subplot(1,3,2)
    plt.title('Chicago',fontsize='small')
    plt.scatter(dest_media[:,1],dest_media[:,0],c=c_list,s=0.1)
    plt.xticks([])
    plt.yticks([])
final_dict={}
for i in id2num.keys():
    final_dict[i]=[dest[id2num[i],0],dest[id2num[i],1],result[id2num[i]]]
with open('cluster_chi.pickle','wb') as file:
    pickle.dump(final_dict,file)

with open("tonumnyr.pickle","rb") as file:
    id2num=pickle.load(file)["id"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint=torch.load("sv_embedding_10_nyr.tar",map_location=device)
embedding=np.array(checkpoint['model_state_dict']['place_embedding.weight'])
cluster_x=embedding
kmeans=KMeans(n_clusters=4).fit(cluster_x)
col_list=["r","g","b","orange","purple","yellow","black","brown"]
with open("census_tract_business_dict_ny.pickle","rb") as file:
    data=pickle.load(file)
data_n={}
for i in data.keys():
    for j in data[i].keys():
        data[i][j]['fips']=int(i)
        data_n[j]=data[i][j]
del(data)
fips_unique=[]
fips_dict={}
for i in data_n.keys():
    if data_n[i]['fips'] not in fips_unique:
        fips_unique.append(data_n[i]['fips'])
        fips_dict[data_n[i]['fips']]=[i]
    else:
        fips_dict[data_n[i]['fips']].append(i)
dest=[None]*len(id2num)
cluster_x=[]
result=[None]*len(id2num)
label=[None]*len(id2num)
for i in id2num.keys():
    if i in fips_unique:
        x=[]
        destin=[[],[]]
        for j in fips_dict[i]:
            destin[0].append(data_n[j]['coordinates']['latitude'])
            destin[1].append(data_n[j]['coordinates']['longitude'])
        destin=np.array(destin).T
        dest[id2num[i]]=np.mean(destin,axis=0)
        result[id2num[i]]=kmeans.labels_[id2num[i]]
        a=True
        if a:
            label[id2num[i]]=0
        else:
            label[id2num[i]]=1
result=np.array(result)
label=np.array(label)
dest=np.array(dest)
c_list=[]
for i in range(result.shape[0]):
    c_list.append(col_list[result[i]])
for j in range(1):
    dest_media=dest[np.where(label==j)]
    result_media=result[np.where(label==j)]
    plt.subplot(1,3,3)
    plt.title('New York',fontsize='small')
    plt.scatter(dest_media[:,1],dest_media[:,0],c=c_list,s=0.1)
    plt.xticks([])
    plt.yticks([])
final_dict={}
for i in id2num.keys():
    final_dict[i]=[dest[id2num[i],0],dest[id2num[i],1],result[id2num[i]]]
with open('cluster_ny.pickle','wb') as file:
    pickle.dump(final_dict,file)

with open("tonumbar.pickle","rb") as file:
    id2num=pickle.load(file)["id"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint=torch.load("sv_embedding_10_bar.tar",map_location=device)
embedding=np.array(checkpoint['model_state_dict']['place_embedding.weight'])
cluster_x=embedding
kmeans=KMeans(n_clusters=4).fit(cluster_x)
col_list=["r","g","b","orange","purple","yellow","black","brown"]
with open("census_tract_business_dict.pickle","rb") as file:
    data=pickle.load(file)
data_n={}
for i in data.keys():
    for j in data[i].keys():
        data[i][j]['fips']=int(i)
        data_n[j]=data[i][j]
del(data)
fips_unique=[]
fips_dict={}
for i in data_n.keys():
    if data_n[i]['fips'] not in fips_unique:
        fips_unique.append(data_n[i]['fips'])
        fips_dict[data_n[i]['fips']]=[i]
    else:
        fips_dict[data_n[i]['fips']].append(i)
dest=[None]*len(id2num)
cluster_x=[]
result=[None]*len(id2num)
label=[None]*len(id2num)
for i in id2num.keys():
    if i in fips_unique:
        x=[]
        destin=[[],[]]
        for j in fips_dict[i]:
            destin[0].append(data_n[j]['coordinates']['latitude'])
            destin[1].append(data_n[j]['coordinates']['longitude'])
        destin=np.array(destin).T
        dest[id2num[i]]=np.mean(destin,axis=0)
        result[id2num[i]]=kmeans.labels_[id2num[i]]
        a=True
        if a:
            label[id2num[i]]=0
        else:
            label[id2num[i]]=1
result=np.array(result)
label=np.array(label)
dest=np.array(dest)
c_list=[]
for i in range(result.shape[0]):
    c_list.append(col_list[result[i]])
for j in range(1):
    dest_media=dest[np.where(label==j)]
    result_media=result[np.where(label==j)]
    plt.subplot(1,3,1)
    plt.title('The Bay Area',fontsize='small')
    plt.scatter(dest_media[:,1],dest_media[:,0],c=c_list,s=0.1)    
    plt.xticks([])
    plt.yticks([])
final_dict={}
for i in id2num.keys():
    final_dict[i]=[dest[id2num[i],0],dest[id2num[i],1],result[id2num[i]]]
with open('cluster_ba.pickle','wb') as file:
    pickle.dump(final_dict,file)
plt.show()