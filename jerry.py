import pandas as pd
import numpy as np
import os
import datetime
import time
from os import listdir
from os.path import isfile, join


# thanks Wang Yu for the reading methods
def read_district(filename):
    cols = ['district_hash', 'district_id']
    df = pd.read_csv(filename, header=None, sep='\t', names=cols)
    return df


def read_weather(filename):
    cols = ['Time', 'Weather', 'temperature', 'PM2.5']
    df = pd.read_csv(filename, header=None, sep='\t', names=cols)
    return df


def read_traffic(filename):
    cols = ['district_hash', 'tj_level', 'tj_time']
    df = pd.read_csv(filename, header=None, sep='\t', names=cols)
    return df


def read_poi(filename):
    cols = ['district_hash', 'poi_class']
    df = pd.read_csv(filename, header=None, sep='\t', names=cols)
    return df


def read_order(filename):
    cols = ['order_id', 'driver_id', 'passenger_id', 'start_district_hash', 'dest_district_hash', 'price', 'time']
    df = pd.read_csv(filename, header=None, sep='\t', names=cols)
    return df


# concatenating all the days and reading all 5 data sources into dataframes
pathdata = r"..\season_1\training_data\order_data"
files = [f for f in listdir(pathdata) if isfile(join(pathdata, f))]
df_order = pd.DataFrame(index=range(0, 4), columns=['order_id', 'driver_id', 'passenger_id', 'start_district_hash', 'dest_district_hash', 'price', 'time'], dtype='float')
for file in files:
    if file[0][0] != '.':
        df0 = read_order(os.path.join(pathdata, file))
        frames = [df_order, df0]
        df_order = pd.concat(frames, ignore_index=True)
df_order.to_csv(os.path.join(pathdata, 'order'))


pathdata = r"..\season_1\training_data\traffic_data"
files = [f for f in listdir(pathdata) if isfile(join(pathdata, f))]
df_traffic = pd.DataFrame(index=range(0, 4), columns=['district_hash', 'tj_level', 'tj_time'], dtype='float')
for file in files:
    if file[0][0] != '.':
        df0 = read_order(os.path.join(pathdata, file))
        frames = [df_order, df0]
        df_traffic = pd.concat(frames, ignore_index=True)
df_traffic.to_csv(os.path.join(pathdata, 'traffic'))


pathdata = r"..\season_1\training_data\weather_data"
files = [f for f in listdir(pathdata) if isfile(join(pathdata, f))]
df_weather = pd.DataFrame(index=range(0, 4), columns=['Time', 'Weather', 'temperature', 'PM2.5'], dtype='float')
for file in files:
    if file[0][0] != '.':
        df0 = read_order(os.path.join(pathdata, file))
        frames = [df_order, df0]
        df_weather = pd.concat(frames, ignore_index=True)
df_weather.to_csv(os.path.join(pathdata, 'weather'))


pathdata = r"..\season_1\training_data\cluster_map"
df_district = read_district(os.path.join(pathdata, 'cluster_map'))

pathdata = r"..\season_1\training_data\poi_data"
df_poi = read_district(os.path.join(pathdata, 'poi_data'))


# left joining all the dataframes to form an integrated table for Tableau exploration
df = pd.merge(left=df_order, right=df_district, how='left', left_on='start_district_hash', right_on='district_hash')
df = df.rename(columns={'district_id': 'start_district_id'})
df = pd.merge(left=df, right=df_district, how='left', left_on='dest_district_hash', right_on='district_hash')
df = df.rename(columns={'district_id': 'dest_district_id'})

df = pd.merge(left=df, right=df_poi, how='left', left_on='start_district_hash', right_on='district_hash')
df = df.rename(columns={'poi_class': 'start_poi_class'})
df = pd.merge(left=df, right=df_poi, how='left', left_on='dest_district_hash', right_on='district_hash')
df = df.rename(columns={'poi_class': 'dest_poi_class'})

df = pd.merge(left=df, right=df_traffic, how='left', left_on='start_district_hash', right_on='district_hash')
df = df.rename(columns={'tj_level': 'start_tj_level'})
df = pd.merge(left=df, right=df_traffic, how='left', left_on='dest_district_hash', right_on='district_hash')
df = df.rename(columns={'tj_time': 'dest_tj_time'})

df = pd.merge(left=df, right=df_weather, how='left', left_on='start_district_hash', right_on='district_hash')
df = df.rename(columns={'tj_level': 'start_tj_level'})
df = pd.merge(left=df, right=df_weather, how='left', left_on='dest_district_hash', right_on='district_hash')
df = df.rename(columns={'tj_time': 'dest_tj_time'})


e, l = np.unique(df_weather['Time'], return_index=True)
e = np.r_[-np.inf, e + np.ediff1d(e, to_end=np.inf)/2]
df['temp'] = pd.cut(df['start_tj_level'], bins=e, labels=df_weather.index[l])
df.join(df_weather, on='temp', rsuffix='_2')

df = df.drop('start_district_hash')


