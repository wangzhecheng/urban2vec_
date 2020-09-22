# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:38:18 2019

@author: lhy
"""

import pickle
import re
import numpy as np
stop_word=[]
with open('street_view_ba.pickle','rb') as file:
    street=pickle.load(file)
    street_fips=street['fips']
with open('census_tract_business_dict_ba.pickle','rb') as file:
    census=pickle.load(file)
census_n={}
for i in census.keys():
    if int(i) in street_fips:
        for j in census[i].keys():
            census_n[j]=census[i][j]
            census_n[j]['fips'] = int(i)
del(census)
del(street)
del(street_fips)
with open('stop_word.txt','r',encoding='utf-8') as file:
    for row in file:
        stop_word.append(row.strip())
"""
with open('reviews84448.pickle','rb') as file:
    review=pickle.load(file)
review_l={}
for i in census_n.keys():
    if i in review.keys():
        media_set=set()
        for j in review[i]:
            media=j['text'].lower().strip()
            media1=re.split('[,.\s!&#:_()\-$"\'\+\*%^\[\]@?\=\;/\\~`|0123456789]',media)
            for k in media1:
                if k not in stop_word and len(k)>1:
                    media_set.add(k)
        review_l[i]=list(media_set)
word_count={}
word_list=[]
for i in review_l.keys():
    for j in review_l[i]:
        if j not in word_count.keys():
            word_count[j]=0
        word_count[j]+=1
for i in word_count.keys():
    if word_count[i]>6 and word_count[i]<1500:
        word_list.append(i)
all_pair=[]
for i in review_l.keys():
    for j in review_l[i]:
        if j in word_list:
            all_pair.append([j,census_n[i]['fips']])

with open('review_pair_ba.pickle','wb') as file:
    pickle.dump(all_pair,file)
with open('word_list_ba.pickle','wb') as file:
    pickle.dump(word_list,file)
"""