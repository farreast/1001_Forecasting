#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:39:22 2020

@author: farrisatif
"""

import numpy as np
import pandas as pd

dfs = [pd.read_csv('solar_gen1.csv',index_col='datetime_beginning_utc',parse_dates=True),
       pd.read_csv('solar_gen2.csv',index_col='datetime_beginning_utc',parse_dates=True)]     
df11,df22 = dfs

def clean_dataframe(df1):
    store = []
    regions = ['MIDATL','SOUTH','WEST']
    for i in range(3):
        df1m = df1[df1['area'] == regions[i]]
        df1m = df1m.rename(columns={'solar_generation_mw': regions[i]+'_solar'})
        df1m = df1m.drop(['area'], axis=1)
        if i > 0:
            df1m = df1m.drop(['datetime_beginning_ept'], axis=1)
        store.append(df1m)
    store = pd.concat(store, axis=1)
    return store

b = clean_dataframe(df11)
c = clean_dataframe(df22)
final = pd.concat([b,c])
final.to_excel("Solar_GenX.xlsx")  




#g = pd.concat([df1m, df1s], axis=1)





#for i in range(5):
#     dfs[i] = dfs[i][['datetime_beginning_ept','fuel_type','mw']]
#     dfs[i] = dfs[i].sort_index()
#     dfs[i] = dfs[i][dfs[i]['fuel_type'] == 'Solar']
#     dfs[i].drop(dfs[i].tail(1).index,inplace=True)

# df1,df2,df3,df4,df5 = dfs
# result = pd.concat(dfs)

# result.to_excel("output.xlsx")  










#df1, df2, df3, df4 = dfs
# df1 = df1[['fuel_type','mw']]
# df2 = df2[['fuel_type','mw']]
# df3 = df3[['fuel_type','mw']]
# df4 = df4[['fuel_type','mw']]

# df1 = df1.sort_index()
# df2 = df2.sort_index()
# df3 = df3.sort_index()
# df4 = df4.sort_index()


# df1 = df1[df1['fuel_type'] == 'Solar']
# df2 = df2[df2['fuel_type'] == 'Solar']
# df3 = df3[df3['fuel_type'] == 'Solar']
# df4 = df4[df4['fuel_type'] == 'Solar']

# df1.drop(df1.tail(1).index,inplace=True)
# df2.drop(df2.tail(1).index,inplace=True)
# df3.drop(df3.tail(1).index,inplace=True)
# df4.drop(df4.tail(1).index,inplace=True)

# frames = [df1 ,df2 ,df3 ,df4]
# result = pd.concat(frames)

















