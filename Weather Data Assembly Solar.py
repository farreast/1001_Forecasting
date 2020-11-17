import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
######################################################################################### 
cincinnati = pd.read_csv('cincinnati_data.csv').set_index('DATE').fillna(method='ffill')
baltimore = pd.read_csv('baltimore_data.csv').set_index('DATE').fillna(method='ffill')
richmond = pd.read_csv('richmond_data.csv').set_index('DATE').fillna(method='ffill')
cincinnati.index = pd.to_datetime(cincinnati.index).round('h')
baltimore.index = pd.to_datetime(baltimore.index).round('h')
richmond.index = pd.to_datetime(richmond.index).round('h')

columns = []
for i in cincinnati.columns:
    if 'Hourly' in i:
        columns.append(i)
        
cincinnati = cincinnati[columns]
baltimore = baltimore[columns]
richmond = richmond[columns]


columns.remove('HourlySkyConditions')
columns.remove('HourlyPresentWeatherType')
columns.remove('HourlyPrecipitation')

for col in columns:
    for i in enumerate(cincinnati[col]):
        if type(i[1]) == int or type(i[1]) == float:
            pass
        elif 's' in i[1]:
            cincinnati[col][cincinnati[col]==i[1]] = float(i[1][:-1])
            
for col in columns:
    for i in enumerate(baltimore[col]):
        if type(i[1]) == int or type(i[1]) == float:
            pass
        elif 's' in i[1]:
            baltimore[col][baltimore[col]==i[1]] = float(i[1][:-1])
            
for col in columns:
    for i in enumerate(richmond[col]):
        if type(i[1]) == int or type(i[1]) == float:
            pass
        elif 's' in i[1]:
            richmond[col][richmond[col]==i[1]] = float(i[1][:-1])
            
#########################################################################################            
ev = pd.read_excel('Solar_GenX.xlsx').set_index('datetime_beginning_ept').fillna(method='bfill')
ev.index = pd.to_datetime(ev.index)            
#########################################################################################          
demand = pd.read_csv('Demand_data.csv',index_col='datetime_beginning_utc',parse_dates=True)
ev1 = ev.iloc[:len(demand),[0,1,2,3]]
ixx = ev1.index.to_list()
demand['datetime_beginning_ept'] = pd.Series(ixx, index=demand.index)
demand = demand.set_index('datetime_beginning_ept')
######################################################################################### 
WIND = pd.read_excel('WindGenDataFinal.xlsx',index_col='datetime_beginning_ept',parse_dates=True)
WIND = WIND['2019-01-01 00:00:00':]
demand = demand['2019-01-01 00:00:00':'2020-11-08 18:00:00']
ev1= ev1['2019-01-01 00:00:00':'2020-11-08 18:00:00']
######################################################################################### 



#g = pd.concat([ev1.reset_index(), df4.reset_index()], axis=1)

# for col in df.columns:
#     df[col] = (pd.to_numeric(df[col], errors='coerce').fillna(0))





