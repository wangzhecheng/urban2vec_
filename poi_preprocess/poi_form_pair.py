# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:14:04 2019

@author: lhy
"""

import pickle
import numpy as np
import json
#from gensim.models import KeyedVectors
import torch
from embedding import Embedding
#wv=KeyedVectors.load('wordba.kv')
with open('word_list_ba.pickle','rb') as file:
    word_list=pickle.load(file)
glove_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
embedding1 = Embedding(glove_url, 200, vocab=set(word_list))
with open('street_view_ba50.pickle','rb') as file:
    street=pickle.load(file)
    embedding=street['x']
    street_fips=street['fips']
with open('census_tract_business_dict_ba.pickle','rb') as file:
    census=pickle.load(file)
with open("categories.json","rb") as file:
    cat=json.load(file)
def digging(leaf,result_list,cat):
    result_list.append(leaf)
    for i in range(len(cat)):
       if cat[i]['alias'] == leaf:
           for j in range(len(cat[i]['parents'])):
               digging(cat[i]['parents'][j],result_list,cat)
           break
    return result_list
yelp_fips=[]
for i in census.keys():
    yelp_fips.append(int(i))
embedding_n=[]
fips_n=[]
for i in range(len(street_fips)):
    if street_fips[i] in yelp_fips:
        embedding_n.append(embedding[i])
        fips_n.append(street_fips[i])
embedding_n=np.array(embedding_n)
census_n={}
for i in census.keys():
    if int(i) in fips_n:
        census_n[int(i)]=census[i]
del(census)
del(street)
del(yelp_fips)
del(embedding)
del(street_fips)
id2num={}
for i in range(len(fips_n)):
    id2num[fips_n[i]]=i
cat_diction={}
all_pair=[]
count1=0
for i in census_n.keys():
    for j in census_n[i].keys():
        count1+=1
        if "price" in census_n[i][j].keys():
            all_pair.append(["pr_"+str(len(census_n[i][j]["price"])),i])
        if "rating" in census_n[i][j].keys():
            all_pair.append(["ra_"+str(census_n[i][j]["rating"]),i])
        if "categories" in census_n[i][j].keys():
            for k in range(len(census_n[i][j]["categories"])):
                media=[]
                media = digging(census_n[i][j]["categories"][k]['alias'],media,cat)
                for n in range(len(media)):
                    all_pair.append(["cat_"+media[n],i])
                    if "cat_"+media[n] in cat_diction.keys():
                        cat_diction["cat_"+media[n]]=cat_diction["cat_"+media[n]]+1
                    else:
                        cat_diction["cat_"+media[n]]=1
i=0
while i < len(all_pair):
    if all_pair[i][0] in cat_diction and cat_diction[all_pair[i][0]]<5:
        del(all_pair[i])
    else:
        i=i+1
with open('review_pair_ba.pickle','rb') as file:
    all_pair.extend(pickle.load(file))
word2num={}
count=0
all_pair_num=[]
id_dict={}
for i in all_pair:
    if i[0] not in word2num.keys():
       word2num[i[0]]=count
       count=count+1
    if id2num[i[1]] not in id_dict.keys():
       id_dict[id2num[i[1]]]=[]
    all_pair_num.append([word2num[i[0]],id2num[i[1]]])
    id_dict[id2num[i[1]]].append(word2num[i[0]])
count=np.zeros(len(word2num),dtype='int')
for i in all_pair:
    count[word2num[i[0]]]+=1
word_embedding=[None]*len(word2num)
for i in word2num.keys():
    try:
        word_embedding[word2num[i]]=embedding1[i]
    except:
        word_embedding[word2num[i]]=np.random.normal(loc=0.0,scale=0.1,size=50)
word_embedding=np.array(word_embedding)
doc={'place_embedding.weight':torch.tensor(embedding_n,requires_grad=True),'word_embedding.weight':torch.tensor(word_embedding,requires_grad=True)}
#doc={'place_embedding.weight':torch.tensor(embedding_n,requires_grad=True)}
#doc={'word_embedding.weight':torch.tensor(word_embedding,requires_grad=True)}
"""
alll={'id':id_dict,'word':all_pair_num}
tonum={'word':word2num,'id':id2num}
torch.save(doc,'docba50.tar')
with open("allba50.pickle","wb") as file:
    pickle.dump(alll,file)
with open("countba50.pickle","wb") as file:
    pickle.dump(count,file)
with open("tonumba50.pickle","wb") as file:
    pickle.dump(tonum,file)
"""