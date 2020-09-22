# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:32:50 2019

@author: lhy
"""
# -*- coding: utf-8 -*-

"""

Yelp Fusion API code sample.



This program demonstrates the capability of the Yelp Fusion API

by using the Search API to query for businesses by a search term and location,

and the Business API to query additional information about the top result

from the search query.



Please refer to http://www.yelp.com/developers/v3/documentation for the API

documentation.



This program requires the Python requests library, which you can install via:

`pip install -r requirements.txt`.



Sample usage of the program:

`python sample.py --term="bars" --location="San Francisco, CA"`

"""

import requests
from urllib.parse import quote
import pickle
import time
import os
def request(host, path, api_key, url_params=None):

    """Given your API_KEY, send a GET request to the API.



    Args:

        host (str): The domain host of the API.

        path (str): The path of the API after the domain.

        API_KEY (str): Your API Key.

        url_params (dict): An optional set of query parameters in the request.



    Returns:

        dict: The JSON response from the request.



    Raises:

        HTTPError: An error occurs from the HTTP request.

    """

    url_params = url_params or {}

    url = '{0}{1}'.format(host, quote(path.encode('utf8')))

    headers = {

        'Authorization': 'Bearer %s' % api_key,

    }



    #print(u'Querying {0} ...'.format(url))



    response = requests.request('GET', url, headers=headers, params=url_params)



    return response.json()
with open("corpus_dict.pickle",'rb') as file:
    data=pickle.load(file)
host = "https://api.yelp.com/v3/businesses/"
api_key_list = ['Qy0c8fDBptB3CygzJ1os5q97NrZ5S9dyDfty_ReFlpzdd_sQQ4fRHKhd8U_63HXthlQ40E6K37kbpCBDSyGWzP6XVSBIA2-DDu4qIAwxUwAXJVs-vmqiUSS0pYq2XHYx',
               'x0GGIL4iw_6MrGwht_rd5PnkrKof6bhOHem5t_TkpRK9gUeWktXC3yGZyg7SK10tgdidNgBRzNm_2bG3IakFpsEYa2dLcBO3oAsO99T1dz3dJG75LyxzEJBctOG2XHYx',
               '7Yg43vabqzEgt0_ZR2EsI0PYDz31taoS7-T5G0rl29Pwrq8aabmVDIHHqCLUFqDScEjRg08u17-ZzkuVxWMfjLaanyNlX4B36oiBynbsZs9QhnjaeLu2yVCqv-O2XHYx',
               'y2aSaUgf84lgpbacuzkdWMAcADq2x3UeUpxCd6VuBc6ZhkrTQpmIAXvm-mGPdr3Nu39Ft3LK-Ixl7hdOnRmGl4gXZ4lWZciVMwtcL4UBenDbQ_QmMuh0tMI8-4i3XHYx',
               '4N8JFyXb6hy6dvwC4LtR21WKABKKC4HiqPYEDAoAswS4-MSW4H64QVVQ_zALs_bgt-s0cjFGCEbEWI74tn9Jt-Mixevro5lA-HRSWN9ie2vug2_l6cRfJmccLLhVXXYx',
               'U_-1mIyZEH2NPV_hbwPQDttRLjULhONyUa_nLyVIY_svoG13TJjr-qngHSu-8VAK0K34OAuer4U-XQV58XmhNlOI9IcQveDKxuZv-DKq4lAJo4VWhLSVQYhV6rlVXXYx',
               'COUw6ZJ6uBM7aebGanhegCFWoLW8tFlmjhhqjbHTWt357rcyn5qBRB8p_5_JkSLkTQI2PWY9qW3X4XjFdBasPjdhEqSZzhQhqtC0fheEX6dQDHaumndS-ITsJ7tVXXYx',
               'k1q6kvfsYzJ0p_bBDLWCx_EP5kLWMwpBKxDiRYBHNt_K_vpBgjuNosB1KibzHV24PwyYWwwuQKsoifWsHvAmq-sePrDEvDbUZMNk14GIYrUf4VMdoKRYu0kDz71VXXYx',
               'UspmzjjtuqmGM350viwPWhKiqbjuZf0-uFRaoAsUAFy15UT8763eoQR63f4-3RVxiWvoxpcgLa8fwuOnZZ_5-WlcDUC6fka0jWe5R5P2dvtvLMZ4GbnSOWZjmcBVXXYx',
               'BUrMIheTXt2H1aBB3KkRl1ZmPt3uAqNy2ltMu30BCoFJBK3iWfN6F-d9f7hbVAIL6dZFnRuLUmRTPlLY62eXVpEmOptw9AbtWDpdj7XFxvaQpDYzfPYX3GRne8FVXXYx']
old_ckpt_dir=None
save_point=[5000,10000,15000,20000,25000,30000,35000]
error_list={}
if old_ckpt_dir==None:
    result={}
    unfinished_id=[]
    for i in data.keys():
        unfinished_id.append(i)
else:
    with open(os.path.join(old_ckpt_dir,"reviews10000.pickle"),"rb") as file:
        result=pickle.load(file)
    with open(os.path.join(old_ckpt_dir,"unfinished10000.pickle"),"rb") as file:
        unfinished_id=pickle.load(file)
count=0
error_last=0
start_iter=0
for i in data.keys():
    count=count+1
    if count%100==0:
        print(count)
    if error_last >100:
        print(count)
        print("error:early stopped")
        break
    if i in unfinished_id and count>start_iter:
        key=api_key_list[count%len(api_key_list)]
        path=i+"/reviews"
        try:
            output=request(host,path,key)
        except:
            error_list[i]="request failed"
            error_last=error_last+1
            print("error"+i)
            time.sleep(1)
            continue
        if 'error' in output.keys() or 'reviews' not in output.keys():
            error_list[i]=output
            error_last=error_last+1
            print(output)
            time.sleep(1)
            continue
        result[i]=[]
        unfinished_id.remove(i)
        for j in range(len(output['reviews'])):
            media={}
            media["rating"]=output['reviews'][j]['rating']
            media["text"]=output['reviews'][j]['text']
            media["time"]=output['reviews'][j]['time_created']
            result[i].append(media)
        error_last=0
    if count in save_point and count>start_iter:
        print(count)
        with open("reviews"+str(count)+".pickle","wb") as file:
            pickle.dump(result,file)
        with open("error"+str(count)+".pickle","wb") as file:
            pickle.dump(error_list,file)
        with open("unfinished"+str(count)+".pickle","wb") as file:
            pickle.dump(unfinished_id,file)
with open("reviews"+str(count)+".pickle","wb") as file:
    pickle.dump(result,file)
with open("error"+str(count)+".pickle","wb") as file:
    pickle.dump(error_list,file)
with open("unfinished"+str(count)+".pickle","wb") as file:
    pickle.dump(unfinished_id,file)


        
    


