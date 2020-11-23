# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
# %% Nasser's cleaning
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
            
# %%  Index Dataframes in uniform manner, create general column names


ev = pd.read_excel('Solar_GenX.xlsx').set_index('datetime_beginning_ept').fillna(method='bfill') ## read in solar data
ev.index = pd.to_datetime(ev.index)                                                                 
#########################################################################################          
demand = pd.read_csv('Demand_data.csv',index_col='datetime_beginning_utc',parse_dates=True) ## read in demand data
ev1 = ev.iloc[:len(demand),[0,1,2,3]]                                                        ## make demand and solar df's the same length
ixx = ev1.index.to_list()
demand['datetime_beginning_ept'] = pd.Series(ixx, index=demand.index)     
demand = demand.set_index('datetime_beginning_ept')                      ### demand data had utc time, so converted to ept using solar df index
######################################################################################### 
WIND = pd.read_excel('WindGenDataFinal.xlsx',index_col='datetime_beginning_ept',parse_dates=True) 
WIND = WIND['2019-01-01 00:00:00':] ## start at same time as others
demand = demand['2019-01-01 00:00:00':'2020-11-08 18:00:00'] ## make demand end at same time as wind
ev1= ev1['2019-01-01 00:00:00':'2020-11-08 18:00:00'] ## make solar end at same time as wind
WIND = WIND.rename(columns={"MIDATL": "MIDATL_wind", "SOUTH": "SOUTH_wind","WEST": "WEST_wind"})
demand = demand.rename(columns={"MIDATL_DEMAND": "MIDATL_demand", "SOUTH_DEMAND": "SOUTH_demand","WEST_DEMAND": "WEST_demand"})



# %% Merge & create dataframes on region: WEST, MIDATL, SOUTH

WEST,MIDATL,SOUTH = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
regions = ['WEST','MIDATL','SOUTH']
cities = [cincinnati,baltimore,richmond]
l = [WEST,MIDATL,SOUTH]
for i in range(len(regions)): #for each city or region
    df = pd.merge(cities[i],ev1[str(regions[i])+'_solar'],left_index=True, right_index=True)   # merge weather with solar
    df = pd.merge(df,WIND[str(regions[i])+'_wind'],left_index=True, right_index=True) # merge previous with wind
    df = pd.merge(df,demand[str(regions[i])+'_demand'],left_index=True, right_index=True) # merge previous with demand
    df = df.drop(['HourlyPresentWeatherType', 'HourlySkyConditions', 
                  'HourlyAltimeterSetting', 'HourlyPressureChange', 'HourlyPrecipitation',
                  'HourlyPressureTendency', 'HourlySeaLevelPressure', 'HourlyWetBulbTemperature'], axis=1)
    l[i] = df.reset_index().drop_duplicates(subset='index', keep='last').set_index('index').sort_index()
    for col in l[i].columns:
        l[i][col] = (pd.to_numeric(l[i][col], errors='coerce').fillna(0))

WEST,MIDATL,SOUTH = l
WEST.to_excel("WEST.xlsx")  
MIDATL.to_excel("MIDATL.xlsx")  
SOUTH.to_excel("SOUTH.xlsx")  
# %% <Wind> and solar Analysis
# plt.figure(figsize = (35,25))
# dfs = [WEST,MIDATL,SOUTH]
# regions = ['WEST','MIDATL','SOUTH']
# REN = ['_solar','_wind']
# ranges = [range(1,4),range(4,7)]

# for r in range(len(REN)):
    
#     for i,g in zip(range(len(dfs)) , ranges[r] ):
       
#         ax = plt.subplot(3,3,g)
        
#         X = dfs[i].drop([str(regions[i])+'_solar',str(regions[i])+'_demand',str(regions[i])+'_wind'], axis=1)
#         Y = dfs[i][str(regions[i])+REN[r]]
        
#         X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2, shuffle = False)
        
#         dt = DecisionTreeRegressor(criterion = 'mse')
#         lr = LinearRegression()
        
#         dt.fit(X_train,y_train)
#         lr.fit(X_train,y_train)
        
#         y_1 = dt.predict(X_test)
#         y_2 = lr.predict(X_test)
        
        
#         ax.plot(X_test.index.tolist(), y_test,color = 'midnightblue',label="Actual")
#         ax.plot(X_test.index.tolist(), y_1, color="red",alpha=0.5,
#                   label="DT prediction", linewidth=2)
#         ax.plot(X_test.index.tolist(), y_2, color="skyblue", alpha=0.7,
#                   label="Linear Reg predcition", linewidth=2)
        
#         labels = ax.get_xticks()[::2]
#         ax.set_xticks(labels)
#         plt.legend()
#         if REN[r] == '_solar':
#             plt.title('SOLAR : {}'.format(regions[i]))
#         if REN[r] == '_wind':
#             plt.title('WIND : {}'.format(regions[i]))
            
            
# %% Solar Analysis  
# dfs = [WEST,MIDATL,SOUTH]
# regions = ['WEST','MIDATL','SOUTH']
# plt.figure(figsize = (25,17))

# for i,g in zip(range(len(dfs)) ,range(1,4) ):
#     ax = plt.subplot(3,3,g)
    
#     X = dfs[i].drop([str(regions[i])+'_solar',str(regions[i])+'_demand',str(regions[i])+'_wind'], axis=1)
#     Y = dfs[i][str(regions[i])+'_solar']
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2, shuffle = False)
    
#     dt = DecisionTreeRegressor(criterion = 'mse')
#     lr = LinearRegression()
    
#     dt.fit(X_train,y_train)
#     lr.fit(X_train,y_train)
    
#     y_1 = dt.predict(X_test)
#     y_2 = lr.predict(X_test)
    
#     ax.plot(X_test.index.tolist(), y_test,color = 'midnightblue',label="Actual")
#     ax.plot(X_test.index.tolist(), y_1, color="red",alpha=0.5,
#               label="DT prediction", linewidth=2)
#     ax.plot(X_test.index.tolist(), y_2, color="skyblue", alpha=0.7,
#               label="Linear Reg predcition", linewidth=2)
#     ax.set_xticks(ax.get_xticks()[::2])
#     plt.legend()
#     plt.title('Solar_region : {}'.format(regions[i]))

# %% <Wind> Analysis


# dfs = [WEST,MIDATL,SOUTH]
# regions = ['WEST','MIDATL','SOUTH']
# plt.figure(figsize = (35,25))
# for i,g in zip(range(len(dfs)) ,range(1,4) ):
#     ax = plt.subplot(3,3,g)
    
#     X = dfs[i].drop([str(regions[i])+'_solar',str(regions[i])+'_demand',str(regions[i])+'_wind'], axis=1)
#     Y = dfs[i][str(regions[i])+'_wind']
    
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2, shuffle = False)
    
#     dt = DecisionTreeRegressor(criterion = 'mse')
#     lr = LinearRegression()
    
#     dt.fit(X_train,y_train)
#     lr.fit(X_train,y_train)
    
#     y_1 = dt.predict(X_test)
#     y_2 = lr.predict(X_test)
    
    
#     ax.plot(X_test.index.tolist(), y_test,color = 'midnightblue',label="Actual")
#     ax.plot(X_test.index.tolist(), y_1, color="red",alpha=0.5,
#               label="DT prediction", linewidth=2)
#     ax.plot(X_test.index.tolist(), y_2, color="skyblue", alpha=0.7,
#               label="Linear Reg predcition", linewidth=2)
    
#     labels = ax.get_xticks()[::2]
#     ax.set_xticks(labels)
#     plt.legend()
#     plt.title('WIND_region : {}'.format(regions[i]))











