# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 12:22:45 2019

@author: lhy
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import pickle
import csv
from sklearn.preprocessing import scale
import torch

with open("embeddingny_chi1.pickle","rb") as file:
    u = pickle._Unpickler(file)
    u.encoding = 'latin1'
    embedding = u.load()
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint=torch.load("sv_embedding_20_last.tar",map_location=device)
embedding=np.array(checkpoint['model_state_dict']['place_embeddding.weight'])
"""
with open("fipsny_chi1.pickle","rb") as file:
    fips=pickle.load(file)
with open('supervised1_chi1_split0_fips_split.pickle','rb') as file:
    split=pickle.load(file)
with open("census_tract_business_dict_chi.pickle","rb") as file:
    data1=pickle.load(file)
data=[]
for i in data1.keys():
    data.append(int(i))
census=[]
census_index=[]
keyss=[]
rm_list=[0,9,10,15,16,19,20,21,22,23,24,25]
with open("census_chicago_sub.csv") as file:
    countt=0
    f_csv=csv.reader(file)
    for row in f_csv:
        if countt==0:
            for k in range(28):
                if k not in rm_list:
                    keyss.append(row[k])
            countt=countt+1
        else:
            if int(row[28]) in fips and data:
                media=[]
                for k in range(28):
                    if k not in rm_list:
                        if row[k] == "" :
                            media.append(None)
                        else:
                            media.append(float(row[k]))
                census_index.append(int(row[28]))    
                census.append(media)
for i in range(len(census[0])):
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
fips_unique=[]
image_unique={}
count2=0
a=1
for i in range(len(fips)):
    if a:
        if fips[i] in fips_unique:
            count2=count2+1
            image_unique[fips[i]].append(i)
        else:
            image_unique[fips[i]]=[i]
            count2=count2+1
            fips_unique.append(fips[i])
result={}
for i in range(len(census[0])):
    result[keyss[i]]=[]
for k in range(20):
    np.random.seed(k)
    np.random.shuffle(fips_unique)
    train_fips=fips_unique[0:int(0.7*len(fips_unique))]
    test_fips=fips_unique[int(0.85*len(fips_unique)):len(fips_unique)]
    #train_fips=split['train']
    #test_fips=split['test']
    train_x=[]
    for i in range(len(train_fips)):
        train_x.append(np.mean(embedding[image_unique[train_fips[i]]],axis=0))
    train_x=np.array(train_x)
    #pca=PCA(n_components=50)
    #pca.fit(train_x)
    #train_x=pca.transform(train_x)
    test_x=[]
    for i in range(len(test_fips)):
        test_x.append(np.mean(embedding[image_unique[test_fips[i]]],axis=0))
    test_x=np.array(test_x)
    #test_x=pca.transform(test_x)
    test_y=[]
    for i in range(len(test_fips)):
        for j in range(len(census_index)):
            if test_fips[i]==census_index[j]:
                test_y.append(census[j])
    test_y=np.array(test_y)
    train_y=[]
    for i in range(len(train_fips)):
        for j in range(len(census_index)):
            if train_fips[i]==census_index[j]:
                train_y.append(census[j])
    train_y=np.array(train_y)
    for i in range(len(census[0])):
        #rfr=LinearRegression()
        rfr=SVR(gamma='auto')
        reg=rfr.fit(train_x,train_y[:,i])
        y_pred=reg.predict(test_x)
        result[keyss[i]].append(r2_score(test_y[:,i],y_pred))
row=[]
for i in range(len(census[0])):
    print(keyss[i])
    row.append(np.mean(result[keyss[i]]))
    print(np.mean(result[keyss[i]]))
print(np.mean(np.array(row)))
row.append(np.mean(np.array(row)))
with open('result.csv','w',newline='') as file:
    csvw=csv.writer(file)
    csvw.writerow(keyss)
    csvw.writerow(row)
"""
x=np.vstack((train_x,test_x))
train_fips.extend(test_fips)
with open('street_view_chisl.pickle','wb') as file:
    pickle.dump({'x':x,'fips':train_fips},file)
"""
