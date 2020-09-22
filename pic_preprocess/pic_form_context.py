# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:25:36 2019

@author: lhy
"""

import pickle
import os
import numpy as np
import csv
from geopy.distance import great_circle
import tqdm
data_dir="meta_data/image_meta"
data=[]
for i in range(1200):
    media_dir=os.path.join(data_dir,str(i)+".csv")
    with open(media_dir,"r") as file:
        csv_r=csv.reader(file)
        count=0
        for row in csv_r:
            if count==0:
                count=count+1
            else:
                media=[str(i)+"/"+str(row[0])+".png",float(row[5]),float(row[6]),int(row[2])]
                data.append(media)
context={}
count=0
for i in range(len(data)):
    context[data[i][0]]=[]
    print('i:'+str(i))
    dist=np.zeros(len(data))
    for j in range(len(data)):
        lat1=data[i][1]
        lon1=data[i][2]
        lat2=data[j][1]
        lon2=data[j][2]
        dist[j]=great_circle((lat1,lon1),(lat2,lon2)).km
    rank=np.argpartition(dist,6)[0:6]
    context[data[i][0]]=[]
    for j in range(rank.shape[0]):
        if rank[j] != i: 
            context[data[i][0]].append(data[rank[j]][0])
with open("context_knn.pickle","wb") as file:
    pickle.dump(context,file)
with open("data_knn.pickle","wb") as file:
    pickle.dump(data,file)
