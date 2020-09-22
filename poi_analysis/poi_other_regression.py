# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 22:30:58 2019

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

with open("tonumbar.pickle","rb") as file:
    id2num=pickle.load(file)["id"]
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint=torch.load("sv_embedding_10_bar.tar",map_location=device)
embedding=np.array(checkpoint['model_state_dict']['place_embedding.weight'])
del(checkpoint)
"""
"""
with open("street_view_basl+poi.pickle",'rb') as file:
    embedding=pickle.load(file)
with open('supervised1_chi1_split0_fips_split.pickle','rb') as file:
    split=pickle.load(file)
"""
solarx=[]
solarx_index=[]
keyxs=[]
x_list=['average_household_size',
 'housing_unit_median_gross_rent',
 'housing_unit_median_value',
 'average_household_income',
 'gini_index',
 'population_density',
 'poverty_family_below_poverty_level_rate',
 'employ_rate',
 'race_asian_rate',
 'race_black_africa_rate',
 'race_white_rate',
 'race_indian_alaska_rate',
 'race_islander_rate',
 'race_two_more_rate',
 'diversity',
 'heating_fuel_coal_coke_rate',
 'heating_fuel_electricity_rate',
 'heating_fuel_solar_rate',
 'heating_fuel_fuel_oil_kerosene_rate',
 'heating_fuel_gas_rate',
 'heating_fuel_none_rate',
 'electricity_price_residential',
 'voting_2016_dem_percentage',
 'voting_2016_gop_percentage',
 'education_less_than_high_school_rate',
 'education_high_school_graduate_rate',
 'education_college_rate',
 'education_bachelor_rate',
 'education_master_rate',
 'education_professional_school_rate',
 'education_doctoral_rate',
 'age_18_24_rate',
 'age_25_34_rate',
 'age_more_than_85_rate',
 'age_75_84_rate',
 'age_35_44_rate',
 'age_45_54_rate',
 'age_65_74_rate',
 'age_55_64_rate',
 'age_10_14_rate',
 'age_15_17_rate',
 'age_5_9_rate',
 'household_type_family_rate',
 'dropout_16_19_inschool_rate',
 'occupation_construction_rate',
 'occupation_public_rate',
 'occupation_information_rate',
 'occupation_finance_rate',
 'occupation_education_rate',
 'occupation_administrative_rate',
 'occupation_manufacturing_rate',
 'occupation_wholesale_rate',
 'occupation_retail_rate',
 'occupation_transportation_rate',
 'occupation_arts_rate',
 'occupation_agriculture_rate',
 'occupancy_vacant_rate',
 'occupancy_owner_rate',
 'mortgage_with_rate',
 'transportation_home_rate',
 'transportation_car_alone_rate',
 'transportation_walk_rate',
 'transportation_carpool_rate',
 'transportation_motorcycle_rate',
 'transportation_bicycle_rate',
 'transportation_public_rate',
 'travel_time_less_than_10_rate',
 'travel_time_10_19_rate',
 'travel_time_20_29_rate',
 'travel_time_30_39_rate',
 'travel_time_40_59_rate',
 'travel_time_60_89_rate',
 'health_insurance_public_rate',
 'health_insurance_none_rate',
 'age_median',
 'travel_time_average',
 'number_of_years_of_education']
zhenkey=[]
with open("solar_adoption_ba_all.csv") as file:
    countt=0
    f_csv=csv.reader(file)
    for row in f_csv:
        if countt==0:
            for k in range(88):
                keyxs.append(row[k])
            countt=countt+1
        else:
            if int(row[88]) in id2num.keys():
                media=[]
                for k in range(88):
                    if keyxs[k] in x_list:
                        if countt==1:
                            zhenkey.append(keyxs[k])
                        if row[k] == "" :
                            media.append(None)
                        else:
                            media.append(float(row[k]))
                solarx_index.append(int(row[88]))    
                solarx.append(media)
                countt=countt+1
for i in range(77):
    summ=0.0
    count=0.0
    for j in range(len(solarx)):
        if solarx[j][i] is not None:
            count=count+1.0
            summ=summ+solarx[j][i]
    mean=summ/count
    for j in range(len(solarx)):
        if solarx[j][i] is None:
            solarx[j][i]=mean
solarx=scale(np.array(solarx))
embedding=solarx
id2num={}
for i in range(len(solarx_index)):
    id2num[solarx_index[i]]=i

sale={}
keyss=[]
with open("real_estate_sale_ba.csv") as file:
    countt=0
    f_csv=csv.reader(file)
    for row in f_csv:
        if countt==0:
            for k in range(len(row)-1):
                keyss.append(row[k])
            countt=countt+1
        else:
            if int(row[-1]) in id2num.keys():
                media=[]
                for k in range(len(row)-1):
                    if row[k] == "" :
                        media.append(None)
                    else:
                        media.append(float(row[k]))
                sale[int(row[-1])]=media
result={}
for i in range(len(row)-1):
    result[keyss[i]]=[]
for i in range(len(row)-1):
    all_l=[]
    alll_fips=[]
    for j in sale.keys():
        if sale[j][i] is not None:
            alll_fips.append(j)
            all_l.append(sale[j][i])
    all_l=list(scale(np.array(all_l)))
    alll={}
    for j in range(len(alll_fips)):
        alll[alll_fips[j]]=all_l[j]
    for j in range(20):
        np.random.seed(j)
        chou=np.arange(len(alll_fips))
        np.random.shuffle(chou)
        train_fips=[]
        for k in range(int(0.8*len(alll_fips))):
            train_fips.append(alll_fips[chou[k]])
        test_fips=[]
        for k in range(int(0.8*len(alll_fips)),len(alll_fips)):
            test_fips.append(alll_fips[chou[k]])
        """
        train_fips=split['train']
        test_fips=split['test']
        """
        train_x=[]
        for k in train_fips:
            if k in id2num.keys() and k in alll.keys():
                train_x.append(embedding[id2num[k]])
        test_x=[]
        for k in test_fips:
            if k in id2num.keys() and k in alll.keys():
                test_x.append(embedding[id2num[k]])
        train_x=np.array(train_x)
        test_x=np.array(test_x)
        #pca=PCA(n_components=200)
        #pca.fit(train_x)
        #train_x=pca.transform(train_x)
        #test_x=pca.transform(test_x)
        train_y=[]
        for k in train_fips:
            if k in id2num.keys() and k in alll.keys():
                train_y.append(alll[k])
        test_y=[]
        for k in test_fips:
            if k in id2num.keys() and k in alll.keys():
                test_y.append(alll[k])
        train_y=np.array(train_y)
        test_y=np.array(test_y)
        rfr=SVR(gamma='auto')
        #rfr=LinearRegression()
        #rfr=MLPRegressor(max_iter=3000,alpha=1,learning_rate_init=0.001,hidden_layer_sizes=(5))
        #rfr=RandomForestRegressor(n_estimators=50,max_features='auto')
        reg=rfr.fit(train_x,train_y)
        y_pred=reg.predict(test_x)
        result[keyss[i]].append(r2_score(test_y,y_pred))    
row1=[]
for i in range(len(row)-1):
    print(keyss[i])
    result[keyss[i]]=np.array(result[keyss[i]])
    row1.append(np.mean(result[keyss[i]]))
    print(np.mean(result[keyss[i]]))
print(np.mean(np.array(row1)))
row1.append(np.mean(np.array(row1)))
with open('result_sale.csv','w',newline='') as file:
    csvw=csv.writer(file)
    csvw.writerow(keyss)
    csvw.writerow(row1)
