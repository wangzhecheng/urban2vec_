# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:07:02 2019

@author: lhy
"""

import pickle
import torch
import numpy as np
from sklearn.preprocessing import scale
import csv
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
with open("tonumchiae.pickle","rb") as file:
    id2num=pickle.load(file)["id"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint=torch.load("sv_embedding_10_chiae.tar",map_location=device)
embedding=np.array(checkpoint['model_state_dict']['place_embedding.weight'])
census=[]
census_index=[]
keyss=[]
rm_list=[0,9,10,15,16,19,20,21,22,23,24,25]
with open('supervised1_ba1_split0_fips_split.pickle','rb') as file:
    split=pickle.load(file)
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
            if int(row[28]) in id2num.keys():
                media=[]
                for k in range(28):
                    if k not in rm_list:
                        if row[k] == "" :
                            media.append(None)
                        else:
                            media.append(float(row[k]))
                census_index.append(int(row[28]))    
                census.append(media)
for i in range(16):
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
row_l=[]
for n in range(1):
    result={}
    train_fips_l=[]
    test_fips_l=[]
    for i in range(16):
        result[keyss[i]]=[]
    for k in range(20):
        """
        train_fips=split['train']
        test_fips=split['test']
        train_index=[]
        for i in train_fips:
            train_index.append(id2num[i])
        test_index=[]
        for i in test_fips:
            test_index.append(id2num[i])
        train_x=embedding[train_index]
        test_x=embedding[test_index]
        train_y=[]
        test_y=[]
        for i in train_fips:
            for j in range(len(census_index)):
                if census_index[j]==i:
                    train_y.append(census[j])
                    break
        for i in test_fips:
            for j in range(len(census_index)):
                if census_index[j]==i:
                    test_y.append(census[j])
        train_y=np.array(train_y)
        test_y=np.array(test_y)
        pca=PCA(n_components=50)
        pca.fit(train_x)
        train_x=pca.transform(train_x)
        test_x=pca.transform(test_x)
        """
        chou=np.arange(len(census_index))
        np.random.seed(k)
        np.random.shuffle(chou)
        train_fips=chou[0:int(0.7*len(census_index))]
        test_fips=chou[int(0.85*len(census_index)):]
        train_fips_media=[]
        test_fips_media=[]
        train_x=embedding[train_fips]
        test_x=embedding[test_fips]
        #pca=PCA(n_components=50)
        #pca.fit(train_x)
        #train_x=pca.transform(train_x)
        #test_x=pca.transform(test_x)
        train_y=[]
        test_y=[]
        for i in range(len(train_fips)):
            for j in id2num.keys():
                if id2num[j]==train_fips[i]:
                    for k in range(len(census_index)):
                        if j==census_index[k]:
                            train_y.append(census[k])
                            train_fips_media.append(j)
                            break
                    break        
        for i in range(len(test_fips)):
            for j in id2num.keys():
                if id2num[j]==test_fips[i]:
                    for k in range(len(census_index)):
                        if j==census_index[k]:
                            test_y.append(census[k])
                            test_fips_media.append(j)
                            break
                    break
        train_y=np.array(train_y)
        test_y=np.array(test_y)
        #train_fips_l.append(train_fips_media)
        #test_fips_l.append(test_fips_media)
        #rfr=MLPRegressor(max_iter=4000,alpha=1,learning_rate_init=0.0001,hidden_layer_sizes=(90))
        #rfr=LinearRegression()
        #rfr=RandomForestRegressor(n_estimators=70,max_features='auto')
        rfr=SVR(gamma='auto')
        for i in range(16):
            reg=rfr.fit(train_x,train_y[:,i])
            y_pred=reg.predict(test_x)
            result[keyss[i]].append(r2_score(test_y[:,i],y_pred))
            """
            print(keyss[i])
            plt.scatter(test_y[:,i],y_pred)
            plt.show()
            """
    row=[]
    for i in range(16):
        print(keyss[i])
        row.append(np.mean(result[keyss[i]]))
        print(np.mean(result[keyss[i]]))
    print(np.mean(np.array(row)))
    row.append(np.mean(np.array(row)))
    row_l.append(row)
with open('result.csv','w',newline='') as file:
    csvw=csv.writer(file)
    csvw.writerow(keyss)
    for i in row_l:
        csvw.writerow(i)
