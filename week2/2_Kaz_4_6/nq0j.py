#!/usr/bin/env python
# coding: utf-8

# In[79]:


# note: update data 2020-04-14
mname = 'nq0j'
path = '.'

# In[80]:

import pandas as pd
import datetime
from datetime import date, timedelta

# new data, run once, then crank up set sizes, iter, folds and evaluate RMSLE
# setup code to build a custom blend x2
# practice using 'final' code etc.

NVAL = 0
# NVAL = 7

# for final day adjustment
# TODAY = datetime.datetime(  *datetime.datetime.today().timetuple()[:3] )
# TODAY = datetime.datetime(2020, 4, 8)
TODAY = datetime.datetime(2020, 4, 15)

BAGS = 4
SET_FRAC = 0.33  # 0.4- 0.5    # 0.364 on 0.1 and 0.5/0.5 drop;
# 0.350 on nodrop and 0.1
# 0.358 at 0.20 no real to drop good features
# 0.1 / 1:
# 0.1 / 2:

DROPS = True

PRIVATE = True
USE_PRIORS = False

SUP_DROP = 0.0
ACTIONS_DROP = 0.0
PLACE_FRACTION = 1.0  # 0.4

# ** FEATURE_DROP = 0.4 # drop random % of features (HIGH!!!, speeds it up)
# ** COUNTRY_DROP = 0.35 # drop random % of countries (20-30pct)
# ** FIRST_DATE_DROP = 0.5 # Date_f must be after a certain date, randomly applied

# FEATURE_DROP_MAX = 0.3
LT_DECAY_MAX = 0.3
LT_DECAY_MIN = -0.4

SINGLE_MODEL = False
MODEL_Y = 'agg_dff'  # 'slope'  # 'slope' or anything else for difference/aggregate log gain

# In[81]:


import pandas as pd
import numpy as np
import os

# In[82]:


from collections import Counter
from random import shuffle
import math

# In[83]:


from scipy.stats.mstats import gmean

# In[ ]:


# In[84]:


import matplotlib.pyplot as plt
import matplotlib as matplotlib
import seaborn as sns



pd.options.display.float_format = '{:.8}'.format

# In[86]:


plt.rcParams["figure.figsize"] = (12, 4.75)



# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"


# In[88]:


# %load_ext line_profiler


# In[89]:


train = pd.read_csv(path + '/train.csv')
test = pd.read_csv(path + '/test.csv')

# In[90]:


# drop last nval days from train
tmax = train.Date.max()
dmax = datetime.datetime.strptime(tmax, '%Y-%m-%d').date()
dtrain = dmax - timedelta(NVAL)
print(dmax, dtrain)
qtrain = train.Date <= dtrain.isoformat()
train = train[qtrain]

len(train)
len(test)
# In[91]:


train.Date.max()

# In[92]:


test_dates = test.Date.unique()
test_dates

# simulate week 1 sort of
test = test[test.Date >= '2020-03-25']
test
# In[93]:


pp = 'public'

# In[94]:


# FINAL_PUBLIC_DATE = datetime.datetime(2020, 4, 8)

if PRIVATE:
    test = test[pd.to_datetime(test.Date) > train.Date.max()]
    # test = test[ pd.to_datetime(test.Date) >  FINAL_PUBLIC_DATE]
    pp = 'private'

# In[95]:


test.Date.unique()

# ### Train Fix

# #### Supplement Missing US Data

# In[96]:


revised = pd.read_csv(path + '/outside_data' +
                      '/covid19_train_data_us_states_before_march_09_new.csv')

revised.Date = pd.to_datetime(revised.Date)
revised.Date = revised.Date.apply(datetime.datetime.strftime, args=('%Y-%m-%d',))
# In[97]:


revised = revised[['Province_State', 'Country_Region', 'Date', 'ConfirmedCases', 'Fatalities']]

# In[98]:


train.tail()

# In[99]:


revised.head()

# In[100]:


train.Date = pd.to_datetime(train.Date)
revised.Date = pd.to_datetime(revised.Date)

# In[101]:


rev_train = pd.merge(train, revised, on=['Province_State', 'Country_Region', 'Date'],
                     suffixes=('', '_r'), how='left')

# In[ ]:


# In[102]:


rev_train[~rev_train.ConfirmedCases_r.isnull()].head()

# In[ ]:


# In[ ]:


# In[103]:


rev_train.ConfirmedCases = np.where((rev_train.ConfirmedCases == 0) & ((rev_train.ConfirmedCases_r > 0)) &
                                    (rev_train.Country_Region == 'US'),

                                    rev_train.ConfirmedCases_r,
                                    rev_train.ConfirmedCases)

# In[104]:


rev_train.Fatalities = np.where(~rev_train.Fatalities_r.isnull() &
                                (rev_train.Fatalities == 0) & ((rev_train.Fatalities_r > 0)) &
                                (rev_train.Country_Region == 'US')
                                ,

                                rev_train.Fatalities_r,
                                rev_train.Fatalities)

# In[105]:


rev_train.drop(columns=['ConfirmedCases_r', 'Fatalities_r'], inplace=True)

# In[106]:


train = rev_train

train[train.Province_State == 'California']
import sys


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key=lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

# ### Oxford Actions Database

# In[107]:


# contain_data = pd.read_excel(path + '/outside_data' +
#                           '/OxCGRT_Download_latest_data.xlsx')

contain_data = pd.read_csv(path + '/outside_data' +
                           #  '/OxCGRT_Download_070420_160027_Full.csv')
                           '/OxCGRT_Download_150420_094053_Full.csv')

# In[108]:


contain_data = contain_data[[c for c in contain_data.columns if
                             not any(z in c for z in ['_Notes', 'Unnamed', 'Confirmed',
                                                      'CountryCode',
                                                      'S8', 'S9', 'S10', 'S11',
                                                      'StringencyIndexForDisplay'])]] \
 \
    # In[109]:

contain_data.rename(columns={'CountryName': "Country"}, inplace=True)

# In[110]:


contain_data.Date = contain_data.Date.astype(str).apply(datetime.datetime.strptime, args=('%Y%m%d',))

# In[ ]:


# In[111]:


contain_data_orig = contain_data.copy()

# In[112]:


contain_data.columns

contain_data.columns
# In[ ]:


# In[113]:


cds = []
for country in contain_data.Country.unique():
    cd = contain_data[contain_data.Country == country]
    cd = cd.fillna(method='ffill').fillna(0)
    cd.StringencyIndex = cd.StringencyIndex.cummax()  # for now
    col_count = cd.shape[1]

    # now do a diff columns
    # and ewms of it
    for col in [c for c in contain_data.columns if 'S' in c]:
        col_diff = cd[col].diff()
        cd[col + "_chg_5d_ewm"] = col_diff.ewm(span=5).mean()
        cd[col + "_chg_20_ewm"] = col_diff.ewm(span=20).mean()

    # stringency
    cd['StringencyIndex_5d_ewm'] = cd.StringencyIndex.ewm(span=5).mean()
    cd['StringencyIndex_20d_ewm'] = cd.StringencyIndex.ewm(span=20).mean()

    cd['S_data_days'] = (cd.Date - cd.Date.min()).dt.days
    for s in [1, 10, 20, 30, 50, ]:
        cd['days_since_Stringency_{}'.format(s)] = np.clip((cd.Date - cd[(cd.StringencyIndex > s)].Date.min()).dt.days,
                                                           0, None)

    cds.append(cd.fillna(0)[['Country', 'Date'] + cd.columns.to_list()[col_count:]])
contain_data = pd.concat(cds)

contain_data.columnsdataset.groupby('Country').S_data_days.max().sort_values(ascending=False)[-30:]
contain_data.StringencyIndex.cummax()
contain_data.groupby('Date').count()[90:]
# In[114]:


contain_data.Date.max()

# In[115]:


contain_data.columns

# In[116]:


contain_data[contain_data.Country == 'Australia']

# In[117]:


contain_data.shape

contain_data.groupby('Country').Date.max()[:50]
# In[118]:


contain_data.Country.replace({'United States': "US",
                              'South Korea': "Korea, South",
                              'Taiwan': "Taiwan*",
                              'Myanmar': "Burma", 'Slovak Republic': "Slovakia",
                              'Czech Republic': 'Czechia',

                              }, inplace=True)

# In[119]:


set(contain_data.Country) - set(test.Country_Region)

# In[ ]:


# #### Load in Supplementary Data

# In[120]:


sup_data = pd.read_excel(path + '/outside_data' +
                         '/Data Join - Copy1.xlsx')

# In[121]:


sup_data.columns = [c.replace(' ', '_') for c in sup_data.columns.to_list()]

# In[122]:


sup_data.drop(columns=[c for c in sup_data.columns.to_list() if 'Unnamed:' in c], inplace=True)

# In[ ]:


# In[ ]:


sup_data.drop(columns=['longitude', 'temperature', 'humidity',
                       'latitude'], inplace=True)
sup_data.columnssup_data.drop(columns=[c for c in sup_data.columns if
                                       any(z in c for z in ['state', 'STATE'])], inplace=True)
sup_data = sup_data[['Province_State', 'Country_Region',
                     'Largest_City',
                     'IQ', 'GDP_region',
                     'TRUE_POPULATION', 'pct_in_largest_city',
                     'Migrant_pct',
                     'Avg_age',
                     'latitude', 'longitude',
                     'abs_latitude',  # 'Personality_uai', 'Personality_ltowvs',
                     'Personality_pdi',

                     'murder', 'real_gdp_growth'
                     ]]
sup_data = sup_data[['Province_State', 'Country_Region',
                     'Largest_City',
                     'IQ', 'GDP_region',
                     'TRUE_POPULATION', 'pct_in_largest_city',
                     # 'Migrant_pct',
                     # 'Avg_age',
                     # 'latitude', 'longitude',
                     #    'abs_latitude', #  'Personality_uai', 'Personality_ltowvs',
                     #   'Personality_pdi',

                     'murder',  # 'real_gdp_growth'
                     ]]
# In[123]:


sup_data.drop(columns=['Date', 'ConfirmedCases',
                       'Fatalities', 'log-cases', 'log-fatalities', 'continent'], inplace=True)

sup_data.drop(columns=['Largest_City',
                       'continent_gdp_pc', 'continent_happiness', 'continent_generosity',
                       'continent_corruption', 'continent_Life_expectancy', 'TRUE_CHINA',
                       'Happiness', 'Logged_GDP_per_capita',
                       'Social_support', 'HDI', 'GDP_pc', 'pc_GDP_PPP', 'Gini',
                       'state_white', 'state_white_asian', 'state_black',
                       'INNOVATIVE_STATE', 'pct_urban', 'Country_pop',

                       ], inplace=True)
sup_data.columns
# In[124]:


sup_data['Migrants_in'] = np.clip(sup_data.Migrants, 0, None)
sup_data['Migrants_out'] = -np.clip(sup_data.Migrants, None, 0)
sup_data.drop(columns='Migrants', inplace=True)

sup_data.loc[:, 'Largest_City'] = np.log(sup_data.Largest_City + 1)
# In[125]:


sup_data.head()

# In[ ]:


# In[126]:


sup_data.shape

sup_data.loc[4][:50]
# In[ ]:


# #### Revise Columns

# In[127]:


train.Date = pd.to_datetime(train.Date)
test.Date = pd.to_datetime(test.Date)
# contain_data.Date = pd.to_datetime(contain_data.Date)


# In[128]:


train.rename(columns={'Country_Region': 'Country'}, inplace=True)
test.rename(columns={'Country_Region': 'Country'}, inplace=True)
sup_data.rename(columns={'Country_Region': 'Country'}, inplace=True)

# In[129]:


train['Place'] = train.Country + train.Province_State.fillna("")
test['Place'] = test.Country + test.Province_State.fillna("")

# In[130]:


sup_data['Place'] = sup_data.Country + sup_data.Province_State.fillna("")

# In[131]:


len(train.Place.unique())

# In[132]:


sup_data = sup_data[
    sup_data.columns.to_list()[2:]]

# In[133]:


sup_data = sup_data.replace('N.A.', np.nan).fillna(-0.5)

# In[134]:


for c in sup_data.columns[:-1]:
    m = sup_data[c].max()  # - sup_data

    if m > 300 and c != 'TRUE_POPULATION':
        print(c)
        sup_data[c] = np.log(sup_data[c] + 1)
        assert sup_data[c].min() > -1

# In[135]:


for c in sup_data.columns[:-1]:
    m = sup_data[c].max()  # - sup_data

    if m > 300:
        print(c)

# In[ ]:


# In[136]:


DEATHS = 'Fatalities'

# In[ ]:


# In[137]:


len(train.Place.unique())

# In[ ]:


# #### Correct Drop-Offs with interpolation

train[(train.ConfirmedCases.shift(1) > train.ConfirmedCases) &
      (train.Place == train.Place.shift(1)) & (train.ConfirmedCases == 0)]
# In[ ]:


# In[138]:


train.ConfirmedCases = np.where(
    (train.ConfirmedCases.shift(1) > train.ConfirmedCases) &
    (train.ConfirmedCases.shift(1) > 0) & (train.ConfirmedCases.shift(-1) > 0) &
    (train.Place == train.Place.shift(1)) & (train.Place == train.Place.shift(-1)) &
    ~train.ConfirmedCases.shift(-1).isnull(),

    np.sqrt(train.ConfirmedCases.shift(1) * train.ConfirmedCases.shift(-1)),

    train.ConfirmedCases)

# In[139]:


train.Fatalities = np.where(
    (train.Fatalities.shift(1) > train.Fatalities) &
    (train.Fatalities.shift(1) > 0) & (train.Fatalities.shift(-1) > 0) &
    (train.Place == train.Place.shift(1)) & (train.Place == train.Place.shift(-1)) &
    ~train.Fatalities.shift(-1).isnull(),

    np.sqrt(train.Fatalities.shift(1) * train.Fatalities.shift(-1)),

    train.Fatalities)

# In[ ]:


# In[140]:


for i in [0, -1]:
    train.ConfirmedCases = np.where(
        (train.ConfirmedCases.shift(2 + i) > train.ConfirmedCases) &
        (train.ConfirmedCases.shift(2 + i) > 0) & (train.ConfirmedCases.shift(-1 + i) > 0) &
        (train.Place == train.Place.shift(2 + i)) & (train.Place == train.Place.shift(-1 + i)) &
        ~train.ConfirmedCases.shift(-1 + i).isnull(),

        np.sqrt(train.ConfirmedCases.shift(2 + i) * train.ConfirmedCases.shift(-1 + i)),

        train.ConfirmedCases)

# In[ ]:


# In[ ]:


# In[141]:


train[train.Place == 'USVirgin Islands'][-10:]

# In[142]:


train[(train.ConfirmedCases.shift(2) > 2 * train.ConfirmedCases) &
      (train.Place == train.Place.shift(2)) & (train.ConfirmedCases < 100000)]

# In[143]:


train[(train.Fatalities.shift(1) > train.Fatalities) &

      (train.Place == train.Place.shift(1)) & (train.Fatalities < 10000)]




train_bk = train.copy()

train.Date.unique()
# In[157]:


# In[158]:


full_train = train.copy()

full_train[full_train.Place == 'USVirgin Islands']
# ### Graphs

# In[159]:


train_c = train[train.Country == 'China']
train_nc = train[train.Country != 'China']
train_us = train[train.Country == 'US']
# train_nc = train[train.Country != 'China']


# In[160]:


def lplot(data, minDate=datetime.datetime(2000, 1, 1),
          columns=['ConfirmedCases', 'Fatalities']):
    data = data[data.Date >= minDate]
    cols = [c for c in data.columns.to_list() if any(z in c for z in
                                                     columns)]
    list.sort(cols)
    prev_col = cols[0][0]
    for col in cols:
        if prev_col[0] != col[0]:
            x = plt.figure();
        prev_col = col

        for place in data.Place.unique():
            data_s = data[data.Place == place]
            x = plt.plot(data_s.Date, data_s[col] + 1, linewidth=0.2);
            plt.yscale('log');


# In[161]:


REAL = datetime.datetime(2020, 2, 10)

# In[162]:


lplot(train_us)

# In[163]:


lplot(train_c)

# In[ ]:


train[train.ConfirmedCases > 0].groupby('Place').min().sort_values('Date').Date[-50:]
# In[ ]:


# In[164]:


lplot(train_nc)

# In[ ]:


lplot(train_nc, REAL)
train[train.Province_State == 'California']
lplot(train_nc[train_nc.Province_State.isnull()])
# In[ ]:


lplot(train_nc[train_nc.Province_State.isnull()], REAL)
lplot(train_nc[~train_nc.Province_State.isnull()], REAL)
lplot(train[train.Country == "US"])
# In[ ]:


# ### Build Dataset

# In[165]:


dataset = train.copy()

# In[166]:


dataset.head()


# In[ ]:


# In[ ]:


# In[ ]:


# ### Create Lagged Growth Rates (4, 7, 12, 20 day rates)

# In[167]:


def rollDates(df, i, preserve=False):
    df = df.copy()
    if preserve:
        df['Date_i'] = df.Date
    df.Date = df.Date + datetime.timedelta(i)
    return df


# In[168]:


WINDOWS = [1, 2, 4, 7, 12, 20, 30]

# In[169]:


for window in WINDOWS:
    csuffix = '_{}d_prior_value'.format(window)

    base = rollDates(dataset, window)
    dataset = pd.merge(dataset, base[['Date', 'Place',
                                      'ConfirmedCases', 'Fatalities']], on=['Date', 'Place'],
                       suffixes=('', csuffix), how='left')
    #     break;
    for c in ['ConfirmedCases', 'Fatalities']:
        dataset[c + csuffix].fillna(0, inplace=True)
        dataset[c + csuffix] = np.log(dataset[c + csuffix] + 1)
        dataset[c + '_{}d_prior_slope'.format(window)] = (np.log(dataset[c] + 1) - dataset[c + csuffix]) / window
        dataset[c + '_{}d_ago_zero'.format(window)] = 1.0 * (dataset[c + csuffix] == 0)

    # In[170]:

for window1 in WINDOWS:
    for window2 in WINDOWS:
        for c in ['ConfirmedCases', 'Fatalities']:
            if window1 * 1.3 < window2 and window1 * 5 > window2:
                dataset[c + '_{}d_{}d_prior_slope_chg'.format(window1, window2)] = dataset[
                                                                                       c + '_{}d_prior_slope'.format(
                                                                                           window1)] - dataset[
                                                                                       c + '_{}d_prior_slope'.format(
                                                                                           window2)]

dataset.tail()
dataset
# #### First Case Etc.

# In[171]:


places = dataset['Place'].drop_duplicates()
places

# In[172]:


first_case = dataset[dataset.ConfirmedCases >= 1].groupby('Place').min().reindex(places)
tenth_case = dataset[dataset.ConfirmedCases >= 10].groupby('Place').min().reindex(places)
hundredth_case = dataset[dataset.ConfirmedCases >= 100].groupby('Place').min().reindex(places)
thousandth_case = dataset[dataset.ConfirmedCases >= 1000].groupby('Place').min().reindex(places)

# In[173]:


tenth_case

# In[174]:


first_fatality = dataset[dataset.Fatalities >= 1].groupby('Place').min().reindex(places)
tenth_fatality = dataset[dataset.Fatalities >= 10].groupby('Place').min().reindex(places)
hundredth_fatality = dataset[dataset.Fatalities >= 100].groupby('Place').min().reindex(places)
thousandth_fatality = dataset[dataset.Fatalities >= 1000].groupby('Place').min().reindex(places)

# np.isinf(dataset.days_since_hundredth_case).sum()
(dataset.Date - hundredth_case.loc[dataset.Place].Date.values).dt.days
# In[175]:


dataset['days_since_first_case'] = np.clip(
    (dataset.Date - first_case.loc[dataset.Place].Date.values).dt.days.fillna(-1), -1, None)
dataset['days_since_tenth_case'] = np.clip(
    (dataset.Date - tenth_case.loc[dataset.Place].Date.values).dt.days.fillna(-1), -1, None)
dataset['days_since_hundredth_case'] = np.clip(
    (dataset.Date - hundredth_case.loc[dataset.Place].Date.values).dt.days.fillna(-1), -1, None)
dataset['days_since_thousandth_case'] = np.clip(
    (dataset.Date - thousandth_case.loc[dataset.Place].Date.values).dt.days.fillna(-1), -1, None)

# In[176]:


dataset['days_since_first_fatality'] = np.clip(
    (dataset.Date - first_fatality.loc[dataset.Place].Date.values).dt.days.fillna(-1), -1, None)
dataset['days_since_tenth_fatality'] = np.clip(
    (dataset.Date - tenth_fatality.loc[dataset.Place].Date.values).dt.days.fillna(-1), -1, None)
dataset['days_since_hundredth_fatality'] = np.clip(
    (dataset.Date - hundredth_fatality.loc[dataset.Place].Date.values).dt.days.fillna(-1), -1, None)
dataset['days_since_thousandth_fatality'] = np.clip(
    (dataset.Date - thousandth_fatality.loc[dataset.Place].Date.values).dt.days.fillna(-1), -1, None)

# In[ ]:


# In[177]:


dataset['case_rate_since_first_case'] = np.clip(
    (np.log(dataset.ConfirmedCases + 1) - np.log(first_case.loc[dataset.Place].ConfirmedCases.fillna(0).values + 1)) / (
                dataset.days_since_first_case + 0.01), 0, 1)
dataset['case_rate_since_tenth_case'] = np.clip(
    (np.log(dataset.ConfirmedCases + 1) - np.log(tenth_case.loc[dataset.Place].ConfirmedCases.fillna(0).values + 1)) / (
                dataset.days_since_tenth_case + 0.01), 0, 1)
dataset['case_rate_since_hundredth_case'] = np.clip((np.log(dataset.ConfirmedCases + 1) - np.log(
    hundredth_case.loc[dataset.Place].ConfirmedCases.fillna(0).values + 1)) / (dataset.days_since_first_case + 0.01), 0,
                                                    1)
dataset['case_rate_since_thousandth_case'] = np.clip((np.log(dataset.ConfirmedCases + 1) - np.log(
    thousandth_case.loc[dataset.Place].ConfirmedCases.fillna(0).values + 1)) / (dataset.days_since_first_case + 0.01),
                                                     0, 1)

# In[178]:


dataset['fatality_rate_since_first_case'] = np.clip(
    (np.log(dataset.Fatalities + 1) - np.log(first_case.loc[dataset.Place].Fatalities.fillna(0).values + 1)) / (
                dataset.days_since_first_case + 0.01), 0, 1)
dataset['fatality_rate_since_tenth_case'] = np.clip(
    (np.log(dataset.Fatalities + 1) - np.log(tenth_case.loc[dataset.Place].Fatalities.fillna(0).values + 1)) / (
                dataset.days_since_first_case + 0.01), 0, 1)
dataset['fatality_rate_since_hundredth_case'] = np.clip(
    (np.log(dataset.Fatalities + 1) - np.log(hundredth_case.loc[dataset.Place].Fatalities.fillna(0).values + 1)) / (
                dataset.days_since_first_case + 0.01), 0, 1)
dataset['fatality_rate_since_thousandth_case'] = np.clip(
    (np.log(dataset.Fatalities + 1) - np.log(thousandth_case.loc[dataset.Place].Fatalities.fillna(0).values + 1)) / (
                dataset.days_since_first_case + 0.01), 0, 1)

# .plot(kind='hist', bins = 150)


# In[179]:


dataset['fatality_rate_since_first_fatality'] = np.clip(
    (np.log(dataset.Fatalities + 1) - np.log(first_fatality.loc[dataset.Place].Fatalities.fillna(0).values + 1)) / (
                dataset.days_since_first_fatality + 0.01), 0, 1)
dataset['fatality_rate_since_tenth_fatality'] = np.clip(
    (np.log(dataset.Fatalities + 1) - np.log(tenth_fatality.loc[dataset.Place].Fatalities.fillna(0).values + 1)) / (
                dataset.days_since_tenth_fatality + 0.01), 0, 1)
dataset['fatality_rate_since_hundredth_fatality'] = np.clip(
    (np.log(dataset.Fatalities + 1) - np.log(hundredth_fatality.loc[dataset.Place].Fatalities.fillna(0).values + 1)) / (
                dataset.days_since_hundredth_fatality + 0.01), 0, 1)
dataset['fatality_rate_since_thousandth_fatality'] = np.clip((np.log(dataset.Fatalities + 1) - np.log(
    thousandth_fatality.loc[dataset.Place].Fatalities.fillna(0).values + 1)) / (
                                                                         dataset.days_since_thousandth_fatality + 0.01),
                                                             0, 1)

# .plot(kind='hist', bins = 150)


# In[ ]:


# In[180]:


dataset['first_case_ConfirmedCases'] = np.log(first_case.loc[dataset.Place].ConfirmedCases.values + 1)
dataset['first_case_Fatalities'] = np.log(first_case.loc[dataset.Place].Fatalities.values + 1)

# In[ ]:


# In[181]:


dataset['first_fatality_ConfirmedCases'] = np.log(
    first_fatality.loc[dataset.Place].ConfirmedCases.fillna(0).values + 1) * (dataset.days_since_first_fatality >= 0)
dataset['first_fatality_Fatalities'] = np.log(first_fatality.loc[dataset.Place].Fatalities.fillna(0).values + 1) * (
            dataset.days_since_first_fatality >= 0)

# In[182]:


dataset['first_fatality_cfr'] = np.where(dataset.days_since_first_fatality < 0,
                                         -8,
                                         (dataset.first_fatality_Fatalities) -
                                         (dataset.first_fatality_ConfirmedCases))

# In[183]:


dataset['first_fatality_lag_vs_first_case'] = np.where(dataset.days_since_first_fatality >= 0,
                                                       dataset.days_since_first_case - dataset.days_since_first_fatality,
                                                       -1)

# In[ ]:


# #### Update Frequency, MAs of Change Rates, etc.

# In[184]:


dataset['case_chg'] = np.clip(np.log(dataset.ConfirmedCases + 1) - np.log(dataset.ConfirmedCases.shift(1) + 1), 0,
                              None).fillna(0)

# In[185]:


dataset['case_chg_ema_3d'] = dataset.case_chg.ewm(span=3).mean() * np.clip(
    (dataset.Date - dataset.Date.min()).dt.days / 3, 0, 1)
dataset['case_chg_ema_10d'] = dataset.case_chg.ewm(span=10).mean() * np.clip(
    (dataset.Date - dataset.Date.min()).dt.days / 10, 0, 1)

# In[186]:


dataset['case_chg_stdev_5d'] = dataset.case_chg.rolling(5).std() * np.clip(
    (dataset.Date - dataset.Date.min()).dt.days / 5, 0, 1)
dataset['case_chg_stdev_15d'] = dataset.case_chg.rolling(15).std() * np.clip(
    (dataset.Date - dataset.Date.min()).dt.days / 15, 0, 1)

dataset['max_case_chg_3d'] = dataset.case_chg.rolling(3).max() \
                             * np.clip((dataset.Date - dataset.Date.min()).dt.days / 3, 0, 1)
dataset['max_case_chg_10d'] = dataset.case_chg.rolling(10).max() \
                              * np.clip((dataset.Date - dataset.Date.min()).dt.days / 10, 0, 1)
# In[187]:


dataset['case_update_pct_3d_ewm'] = (dataset.case_chg > 0).ewm(span=3).mean() * np.power(
    np.clip((dataset.Date - dataset.Date.min()).dt.days / 3, 0, 1), 2)
dataset['case_update_pct_10d_ewm'] = (dataset.case_chg > 0).ewm(span=10).mean() * np.power(
    np.clip((dataset.Date - dataset.Date.min()).dt.days / 10, 0, 1), 2)
dataset['case_update_pct_30d_ewm'] = (dataset.case_chg > 0).ewm(span=30).mean() * np.power(
    np.clip((dataset.Date - dataset.Date.min()).dt.days / 30, 0, 1), 2)

# In[ ]:


# In[188]:


dataset['fatality_chg'] = np.clip(np.log(dataset.Fatalities + 1) - np.log(dataset.Fatalities.shift(1) + 1), 0,
                                  None).fillna(0)

# In[189]:


dataset['fatality_chg_ema_3d'] = dataset.fatality_chg.ewm(span=3).mean() * np.clip(
    (dataset.Date - dataset.Date.min()).dt.days / 33, 0, 1)
dataset['fatality_chg_ema_10d'] = dataset.fatality_chg.ewm(span=10).mean() * np.clip(
    (dataset.Date - dataset.Date.min()).dt.days / 10, 0, 1)

# In[190]:


dataset['fatality_chg_stdev_5d'] = dataset.fatality_chg.rolling(5).std() * np.clip(
    (dataset.Date - dataset.Date.min()).dt.days / 5, 0, 1)
dataset['fatality_chg_stdev_15d'] = dataset.fatality_chg.rolling(15).std() * np.clip(
    (dataset.Date - dataset.Date.min()).dt.days / 15, 0, 1)

# In[191]:


dataset['fatality_update_pct_3d_ewm'] = (dataset.fatality_chg > 0).ewm(span=3).mean() * np.power(
    np.clip((dataset.Date - dataset.Date.min()).dt.days / 3, 0, 1), 2)
dataset['fatality_update_pct_10d_ewm'] = (dataset.fatality_chg > 0).ewm(span=10).mean() * np.power(
    np.clip((dataset.Date - dataset.Date.min()).dt.days / 10, 0, 1), 2)
dataset['fatality_update_pct_30d_ewm'] = (dataset.fatality_chg > 0).ewm(span=30).mean() * np.power(
    np.clip((dataset.Date - dataset.Date.min()).dt.days / 30, 0, 1), 2)

# In[ ]:


# In[ ]:


# In[192]:


dataset.tail()

# In[ ]:


# #### Add Supp Data

# In[193]:


# lag containment data as one week behind
# contain_data.Date = contain_data.Date + datetime.timedelta(7)


# In[194]:


contain_data.Date.max()

# In[195]:


assert set(dataset.Place.unique()) == set(dataset.Place.unique())
dataset = pd.merge(dataset, sup_data, on='Place', how='left', validate='m:1')
dataset = pd.merge(dataset, contain_data, on=['Country', 'Date'], how='left', validate='m:1')

# In[196]:


dataset['log_true_population'] = np.log(dataset.TRUE_POPULATION + 1)

# In[197]:


dataset['ConfirmedCases_percapita'] = np.log(dataset.ConfirmedCases + 1) - np.log(dataset.TRUE_POPULATION + 1)
dataset['Fatalities_percapita'] = np.log(dataset.Fatalities + 1) - np.log(dataset.TRUE_POPULATION + 1)

# In[ ]:


# ##### CFR
np.log(0 + 0.015 / 1)
BLCFR = -4.295015257684252
# In[198]:


# dataset['log_cfr_bad'] = np.log(dataset.Fatalities + 1) - np.log(dataset.ConfirmedCases + 1)
dataset['log_cfr'] = np.log(
    (dataset.Fatalities + np.clip(0.015 * dataset.ConfirmedCases, 0, 0.3)) / (dataset.ConfirmedCases + 0.1))


# In[199]:


def cfr(case, fatality):
    cfr_calc = np.log((fatality + np.clip(0.015 * case, 0, 0.3)) / (case + 0.1))
    #     cfr_calc =np.array(cfr_calc)
    return np.where(np.isnan(cfr_calc) | np.isinf(cfr_calc),
                    BLCFR, cfr_calc)


# In[200]:


BLCFR = np.median(dataset[dataset.ConfirmedCases == 1].log_cfr[::10])
dataset.log_cfr.fillna(BLCFR, inplace=True)
dataset.log_cfr = np.where(dataset.log_cfr.isnull() | np.isinf(dataset.log_cfr),
                           BLCFR, dataset.log_cfr)
BLCFR

# In[201]:


dataset['log_cfr_3d_ewm'] = BLCFR + (dataset.log_cfr - BLCFR).ewm(span=3).mean() * np.power(
    np.clip((dataset.Date - dataset.Date.min()).dt.days / 3, 0, 1), 2)

dataset['log_cfr_8d_ewm'] = BLCFR + (dataset.log_cfr - BLCFR).ewm(span=8).mean() * np.power(
    np.clip((dataset.Date - dataset.Date.min()).dt.days / 8, 0, 1), 2)

dataset['log_cfr_20d_ewm'] = BLCFR + (dataset.log_cfr - BLCFR).ewm(span=20).mean() * np.power(
    np.clip((dataset.Date - dataset.Date.min()).dt.days / 20, 0, 1), 2)

dataset['log_cfr_3d_20d_ewm_crossover'] = dataset.log_cfr_3d_ewm - dataset.log_cfr_20d_ewm

# In[202]:


dataset.drop(columns='log_cfr', inplace=True)

# In[ ]:


# ##### Per Capita vs. World and Similar Countries

# In[203]:


date_totals = dataset.groupby('Date').sum()

# In[204]:


mean_7d_c_slope = dataset.groupby('Date')[['ConfirmedCases_7d_prior_slope']].apply(lambda x:
                                                                                   np.mean(x[x > 0])).ewm(span=3).mean()
mean_7d_f_slope = dataset.groupby('Date')[['Fatalities_7d_prior_slope']].apply(lambda x:
                                                                               np.mean(x[x > 0])).ewm(span=7).mean()

mean_7d_c_slope.plot()
dataset.columns[:100]
mean_7d_c_slope.plot()
date_totals.Fatalities_7d_prior_slope.plot()
date_counts = dataset.groupby('Date').apply(lambda x: x > 0)

date_totals['world_fatalities_chg'] = (np.log(date_totals.Fatalities + 1) \
                                       - np.log(date_totals.Fatalities.shift(1) + 1)) \
    .fillna(method='bfill')
date_totals['world_cases_chg_10d_ewm'] = \
    date_totals.world_cases_chg.ewm(span=10).mean()
date_totals['world_fatalities_chg_10d_ewm'] = \
    date_totals.world_fatalities_chg.ewm(span=10).mean()
dataset['world_cases_chg_10d_ewm'] = \
    date_totals.loc[dataset.Date].world_cases_chg_10d_ewm.values

dataset['world_fatalities_chg_10d_ewm'] = \
    date_totals.loc[dataset.Date].world_fatalities_chg_10d_ewm.values

# In[205]:


dataset['ConfirmedCases_percapita_vs_world'] = np.log(dataset.ConfirmedCases + 1) - np.log(
    dataset.TRUE_POPULATION + 1) - (
                                                       np.log(date_totals.loc[dataset.Date].ConfirmedCases + 1)
                                                       - np.log(date_totals.loc[dataset.Date].TRUE_POPULATION + 1)
                                               ).values

dataset['Fatalities_percapita_vs_world'] = np.log(dataset.Fatalities + 1) - np.log(dataset.TRUE_POPULATION + 1) - (
        np.log(date_totals.loc[dataset.Date].Fatalities + 1)
        - np.log(date_totals.loc[dataset.Date].TRUE_POPULATION + 1)
).values
dataset['cfr_vs_world'] = dataset.log_cfr_3d_ewm - np.log(
    date_totals.loc[dataset.Date].Fatalities / date_totals.loc[dataset.Date].ConfirmedCases).values

# In[206]:


dataset.Fatalities_percapita_vs_world.plot(kind='hist', bins=250);

# In[207]:


dataset.cfr_vs_world.plot(kind='hist', bins=250);

# In[ ]:


# #### Nearby Countries

# In[208]:


cont_date_totals = dataset.groupby(['Date', 'continent_generosity']).sum()

cont_date_totals.iloc[dataset.Date]
# In[209]:


len(dataset)

dataset.columnsdataset.TRUE_POPULATIONdatasetdataset
# In[210]:


dataset['ConfirmedCases_percapita_vs_continent_mean'] = 0
dataset['Fatalities_percapita_vs_continent_mean'] = 0
dataset['ConfirmedCases_percapita_vs_continent_median'] = 0
dataset['Fatalities_percapita_vs_continent_median'] = 0

for cg in dataset.continent_generosity.unique():
    ps = dataset.groupby("Place").last()
    tp = ps[ps.continent_generosity == cg].TRUE_POPULATION.sum()
    print(tp / 1e9)
    for Date in dataset.Date.unique():
        cd = dataset[(dataset.Date == Date) &
                     (dataset.continent_generosity == cg)] \
            [['ConfirmedCases', 'Fatalities', 'TRUE_POPULATION']]
        #         print(cd)
        cmedian = np.median(np.log(cd.ConfirmedCases + 1) - np.log(cd.TRUE_POPULATION + 1))
        cmean = np.log(cd.ConfirmedCases.sum() + 1) - np.log(tp + 1)
        fmedian = np.median(np.log(cd.Fatalities + 1) - np.log(cd.TRUE_POPULATION + 1))
        fmean = np.log(cd.Fatalities.sum() + 1) - np.log(tp + 1)
        cfrmean = cfr(cd.ConfirmedCases.sum(), cd.Fatalities.sum())
        #         print(cmean)

        #         break;

        dataset.loc[(dataset.Date == Date) &
                    (dataset.continent_generosity == cg),
                    'ConfirmedCases_percapita_vs_continent_mean'] = \
            dataset['ConfirmedCases_percapita'] \
            - (cmean)
        dataset.loc[(dataset.Date == Date) &
                    (dataset.continent_generosity == cg),
                    'ConfirmedCases_percapita_vs_continent_median'] = \
            dataset['ConfirmedCases_percapita'] \
            - (cmedian)

        dataset.loc[(dataset.Date == Date) &
                    (dataset.continent_generosity == cg),
                    'Fatalities_percapita_vs_continent_mean'] = \
            dataset['Fatalities_percapita'] \
            - (fmean)
        dataset.loc[(dataset.Date == Date) &
                    (dataset.continent_generosity == cg),
                    'Fatalities_percapita_vs_continent_median'] = \
            dataset['Fatalities_percapita'] \
            - (fmedian)

        dataset.loc[(dataset.Date == Date) &
                    (dataset.continent_generosity == cg),
                    'cfr_vs_continent'] = \
            dataset.log_cfr_3d_ewm \
            - cfrmean
#
#         r.ConfirmedCases
#         r.Fatalities
#         print(continent)


# In[ ]:


dataset[dataset.Country == 'China'][['Place', 'Date',
                                     'ConfirmedCases_percapita_vs_continent_mean',
                                     'Fatalities_percapita_vs_continent_mean']][1000::10]
dataset[['Place', 'Date',
         'cfr_vs_continent']][10000::5]
# In[ ]:


# In[211]:


all_places = dataset[['Place', 'latitude', 'longitude']].drop_duplicates().set_index('Place',
                                                                                     drop=True)
all_places.head()


# In[212]:


def surroundingPlaces(place, d=10):
    dist = (all_places.latitude - all_places.loc[place].latitude) ** 2 + (
                all_places.longitude - all_places.loc[place].longitude) ** 2
    return all_places[dist < d ** 2][1:n + 1]


surroundingPlaces('Afghanistan', 5)


# In[213]:


def nearestPlaces(place, n=10):
    dist = (all_places.latitude - all_places.loc[place].latitude) ** 2 + (
                all_places.longitude - all_places.loc[place].longitude) ** 2
    ranked = np.argsort(dist)
    return all_places.iloc[ranked][1:n + 1]


# In[214]:


nearestPlaces("Angola", 5)

# In[ ]:


dataset.ConfirmedCases_percapita
# In[215]:


dgp = dataset.groupby('Place').last()
for n in [5, 10, 20]:
    #     dataset['ConfirmedCases_percapita_vs_nearest{}'.format(n)] = 0
    #     dataset['Fatalities_percapita_vs_nearest{}'.format(n)] = 0

    for place in dataset.Place.unique():
        nps = nearestPlaces(place, n)
        tp = dgp.loc[nps.index].TRUE_POPULATION.sum()
        #         print(tp)

        dataset.loc[dataset.Place == place,
                    'ratio_population_vs_nearest{}'.format(n)] = \
            np.log(dataset.loc[dataset.Place == place].TRUE_POPULATION.mean() + 1) \
            - np.log(tp + 1)

        #         dataset.loc[dataset.Place==place,
        #                     'avg_distance_to_nearest{}'.format(n)] = \
        #             (dataset.loc[dataset.Place==place].latitude.mean() + 1)\
        #                 - np.log(tp+1)

        nbps = dataset[(dataset.Place.isin(nps.index))].groupby('Date')[['ConfirmedCases', 'Fatalities']].sum()

        nppc = (np.log(nbps.loc[dataset[dataset.Place == place].Date].fillna(0).ConfirmedCases + 1) - np.log(tp + 1))
        nppf = (np.log(nbps.loc[dataset[dataset.Place == place].Date].fillna(0).Fatalities + 1) - np.log(tp + 1))
        npp_cfr = cfr(nbps.loc[dataset[dataset.Place == place].Date].fillna(0).ConfirmedCases,
                      nbps.loc[dataset[dataset.Place == place].Date] \
                      .fillna(0).Fatalities)
        #         print(npp_cfr)
        #         continue;

        dataset.loc[
            (dataset.Place == place),
            'ConfirmedCases_percapita_vs_nearest{}'.format(n)] = \
            dataset[(dataset.Place == place)].ConfirmedCases_percapita \
            - nppc.values
        dataset.loc[
            (dataset.Place == place),
            'Fatalities_percapita_vs_nearest{}'.format(n)] = \
            dataset[(dataset.Place == place)].Fatalities_percapita \
            - nppf.values
        dataset.loc[
            (dataset.Place == place),
            'cfr_vs_nearest{}'.format(n)] = \
            dataset[(dataset.Place == place)].log_cfr_3d_ewm \
            - npp_cfr

        dataset.loc[
            (dataset.Place == place),
            'ConfirmedCases_nearest{}_percapita'.format(n)] = nppc.values
        dataset.loc[
            (dataset.Place == place),
            'Fatalities_nearest{}_percapita'.format(n)] = nppf.values
        dataset.loc[
            (dataset.Place == place),
            'cfr_nearest{}'.format(n)] = npp_cfr

        dataset.loc[
            (dataset.Place == place),
            'ConfirmedCases_nearest{}_10d_slope'.format(n)] = \
            (nppc.ewm(span=1).mean() - nppc.ewm(span=10).mean()).values
        dataset.loc[
            (dataset.Place == place),
            'Fatalities_nearest{}_10d_slope'.format(n)] = \
            (nppf.ewm(span=1).mean() - nppf.ewm(span=10).mean()).values

        npp_cfr_s = pd.Series(npp_cfr)
        dataset.loc[
            (dataset.Place == place),
            'cfr_nearest{}_10d_slope'.format(n)] = \
            (npp_cfr_s.ewm(span=1).mean() \
             - npp_cfr_s.ewm(span=10).mean()).values

#         print(( npp_cfr_s.ewm(span = 1).mean()\
#                                      - npp_cfr_s.ewm(span = 10).mean() ).values)


# In[ ]:


# In[216]:


dgp = dataset.groupby('Place').last()
for d in [5, 10, 20]:
    #     dataset['ConfirmedCases_percapita_vs_nearest{}'.format(n)] = 0
    #     dataset['Fatalities_percapita_vs_nearest{}'.format(n)] = 0

    for place in dataset.Place.unique():
        nps = surroundingPlaces(place, d)
        dataset.loc[dataset.Place == place, 'num_surrounding_places_{}_degrees'.format(d)] = len(nps)

        tp = dgp.loc[nps.index].TRUE_POPULATION.sum()

        dataset.loc[dataset.Place == place,
                    'ratio_population_vs_surrounding_places_{}_degrees'.format(d)] = \
            np.log(dataset.loc[dataset.Place == place].TRUE_POPULATION.mean() + 1) \
            - np.log(tp + 1)

        if len(nps) == 0:
            continue;

        #         print(place)
        #         print(nps)
        #         print(tp)
        nbps = dataset[(dataset.Place.isin(nps.index))].groupby('Date')[['ConfirmedCases', 'Fatalities']].sum()

        #         print(nbps)
        nppc = (np.log(nbps.loc[dataset[dataset.Place == place].Date].fillna(0).ConfirmedCases + 1) - np.log(tp + 1))
        nppf = (np.log(nbps.loc[dataset[dataset.Place == place].Date].fillna(0).Fatalities + 1) - np.log(tp + 1))
        #         break;
        npp_cfr = cfr(nbps.loc[dataset[dataset.Place == place].Date].fillna(0).ConfirmedCases,
                      nbps.loc[dataset[dataset.Place == place].Date] \
                      .fillna(0).Fatalities)
        dataset.loc[
            (dataset.Place == place),
            'ConfirmedCases_percapita_vs_surrounding_places_{}_degrees'.format(d)] = \
            dataset[(dataset.Place == place)].ConfirmedCases_percapita \
            - nppc.values
        dataset.loc[
            (dataset.Place == place),
            'Fatalities_percapita_vs_surrounding_places_{}_degrees'.format(d)] = \
            dataset[(dataset.Place == place)].Fatalities_percapita \
            - nppf.values
        dataset.loc[
            (dataset.Place == place),
            'cfr_vs_surrounding_places_{}_degrees'.format(d)] = \
            dataset[(dataset.Place == place)].log_cfr_3d_ewm \
            - npp_cfr

        dataset.loc[
            (dataset.Place == place),
            'ConfirmedCases_surrounding_places_{}_degrees_percapita'.format(d)] = nppc.values
        dataset.loc[
            (dataset.Place == place),
            'Fatalities_surrounding_places_{}_degrees_percapita'.format(d)] = nppf.values
        dataset.loc[
            (dataset.Place == place),
            'cfr_surrounding_places_{}_degrees'.format(d)] = npp_cfr

        dataset.loc[
            (dataset.Place == place),
            'ConfirmedCases_surrounding_places_{}_degrees_10d_slope'.format(d)] = \
            (nppc.ewm(span=1).mean() - nppc.ewm(span=10).mean()).values
        dataset.loc[
            (dataset.Place == place),
            'Fatalities_surrounding_places_{}_degrees_10d_slope'.format(d)] = \
            (nppf.ewm(span=1).mean() - nppf.ewm(span=10).mean()).values
        npp_cfr_s = pd.Series(npp_cfr)
        dataset.loc[
            (dataset.Place == place),
            'cfr_surrounding_places_{}_degrees_10d_slope'.format(d)] = \
            (npp_cfr_s.ewm(span=1).mean() \
             - npp_cfr_s.ewm(span=10).mean()).values

# In[ ]:


# In[217]:


for col in [c for c in dataset.columns if 'surrounding_places' in c and 'num_sur' not in c]:
    dataset[col] = dataset[col].fillna(0)
    n_col = 'num_surrounding_places_{}_degrees'.format(col.split('degrees')[0].split('_')[-2])

    print(col)
    #     print(n_col)
    dataset[col + "_times_num_places"] = dataset[col] * np.sqrt(dataset[n_col])
#     print('num_surrounding_places_{}_degrees'.format(col.split('degrees')[0][-2:-1]))


# In[218]:


dataset[dataset.Country == 'US'][['Place', 'Date'] + [c for c in dataset.columns if 'ratio_p' in c]][::50]

# In[ ]:


dataset[dataset.Country == "US"].groupby('Place').last() \
    [[c for c in dataset.columns if 'cfr' in c]].iloc[:10, 8:]
# In[ ]:


dataset[dataset.Place == 'USAlabama'][['Place', 'Date'] \
                                      + [c for c in dataset.columns if 'places_5_degree' in c]] \
    [40::5]
# In[ ]:


# In[219]:


dataset.TRUE_POPULATION

# In[220]:


dataset.TRUE_POPULATION.sum()

# In[221]:


dataset.groupby('Date').sum().TRUE_POPULATION

# In[ ]:


dataset[dataset.ConfirmedCases > 0]['log_cfr'].plot(kind='hist', bins=250)
dataset.log_cfr.isnull().sum()
# In[222]:


dataset['first_case_ConfirmedCases_percapita'] = np.log(dataset.first_case_ConfirmedCases + 1) - np.log(
    dataset.TRUE_POPULATION + 1)

dataset['first_case_Fatalities_percapita'] = np.log(dataset.first_case_Fatalities + 1) - np.log(
    dataset.TRUE_POPULATION + 1)

dataset['first_fatality_Fatalities_percapita'] = np.log(dataset.first_fatality_Fatalities + 1) - np.log(
    dataset.TRUE_POPULATION + 1)

dataset['first_fatality_ConfirmedCases_percapita'] = np.log(dataset.first_fatality_ConfirmedCases + 1) - np.log(
    dataset.TRUE_POPULATION + 1)

# In[ ]:


# In[223]:


dataset['days_to_saturation_ConfirmedCases_4d'] = (- np.log(dataset.ConfirmedCases + 1) + np.log(
    dataset.TRUE_POPULATION + 1)) / dataset.ConfirmedCases_4d_prior_slope
dataset['days_to_saturation_ConfirmedCases_7d'] = (- np.log(dataset.ConfirmedCases + 1) + np.log(
    dataset.TRUE_POPULATION + 1)) / dataset.ConfirmedCases_7d_prior_slope

dataset['days_to_saturation_Fatalities_20d_cases'] = (- np.log(dataset.Fatalities + 1) + np.log(
    dataset.TRUE_POPULATION + 1)) / dataset.ConfirmedCases_20d_prior_slope
dataset['days_to_saturation_Fatalities_12d_cases'] = (- np.log(dataset.Fatalities + 1) + np.log(
    dataset.TRUE_POPULATION + 1)) / dataset.ConfirmedCases_12d_prior_slope

# In[224]:


dataset['days_to_3pct_ConfirmedCases_4d'] = (- np.log(dataset.ConfirmedCases + 1) + np.log(
    dataset.TRUE_POPULATION + 1) - 3.5) / dataset.ConfirmedCases_4d_prior_slope
dataset['days_to_3pct_ConfirmedCases_7d'] = (- np.log(dataset.ConfirmedCases + 1) + np.log(
    dataset.TRUE_POPULATION + 1) - 3.5) / dataset.ConfirmedCases_7d_prior_slope

dataset['days_to_0.3pct_Fatalities_20d_cases'] = (- np.log(dataset.Fatalities + 1) + np.log(
    dataset.TRUE_POPULATION + 1) - 5.8) / dataset.ConfirmedCases_20d_prior_slope
dataset['days_to_0.3pct_Fatalities_12d_cases'] = (- np.log(dataset.Fatalities + 1) + np.log(
    dataset.TRUE_POPULATION + 1) - 5.8) / dataset.ConfirmedCases_12d_prior_slope

# In[ ]:


# In[ ]:


# In[225]:


dataset.tail()

# In[ ]:


# ### Build Intervals into Future

# In[ ]:


# In[ ]:


# In[226]:


dataset = dataset[dataset.ConfirmedCases > 0]

len(dataset)

# In[227]:


datas = []
for window in range(1, 35):
    base = rollDates(dataset, window, True)
    datas.append(pd.merge(dataset[['Date', 'Place',
                                   'ConfirmedCases', 'Fatalities']], base, on=['Date', 'Place'],
                          how='right',
                          suffixes=('_f', '')))
data = pd.concat(datas, axis=0).astype(np.float32, errors='ignore')

# In[228]:


len(data)

data[data.Place == 'USNew York']
# In[229]:


data['Date_f'] = data.Date
data.Date = data.Date_i

# In[230]:


data['elapsed'] = (data.Date_f - data.Date_i).dt.days

# In[231]:


data['CaseChgRate'] = (np.log(data.ConfirmedCases_f + 1) - np.log(data.ConfirmedCases + 1)) / data.elapsed;
data['FatalityChgRate'] = (np.log(data.Fatalities_f + 1) - np.log(data.Fatalities + 1)) / data.elapsed;

falloff_hash = {}


def true_agg(rate_i, elapsed, bend_rate):
    #     print(elapsed);
    elapsed = int(elapsed)
    #     ar = 0
    #     rate = rate_i
    #     for i in range(0, elapsed):
    #         rate *= bend_rate
    #         ar += rate
    #     return ar

    if (bend_rate, elapsed) not in falloff_hash:
        falloff_hash[(bend_rate, elapsed)] = np.sum([np.power(bend_rate, e) for e in range(1, elapsed + 1)])
    return falloff_hash[(bend_rate, elapsed)] * rate_i


# In[235]:


true_agg(0.3, 30, 0.9)

true_agg(0.3, 30, 0.9)
# In[236]:


slope_cols = [c for c in data.columns if
              any(z in c for z in ['prior_slope', 'chg', 'rate'])
              and not any(z in c for z in ['bend', 'prior_slope_chg', 'Country', 'ewm',
                                           ])]  # ** bid change; since rate too stationary
print(slope_cols)
bend_rates = [1, 0.95, 0.90]
for bend_rate in bend_rates:
    bend_agg = data[['elapsed']].apply(lambda x: true_agg(1, *x, bend_rate), axis=1)

    for sc in slope_cols:
        if bend_rate < 1:
            data[sc + "_slope_bend_{}".format(bend_rate)] = data[sc] * np.power((bend_rate + 1) / 2, data.elapsed)

            data[sc + "_true_slope_bend_{}".format(bend_rate)] = bend_agg * data[sc] / data.elapsed

        data[sc + "_agg_bend_{}".format(bend_rate)] = data[sc] * data.elapsed * np.power((bend_rate + 1) / 2,
                                                                                         data.elapsed)

        data[sc + "_true_agg_bend_{}".format(bend_rate)] = bend_agg * data[sc]
#                       data[[sc, 'elapsed']].apply(lambda x: true_agg(*x, bend_rate), axis=1)


#         print(data[sc+"_true_agg_bend_{}".format(bend_rate)])

data[[c for c in data.columns if 'Fatalities_7d_prior_slope' in c and 'true_agg' in c]]
# In[ ]:


# In[ ]:


data[data.Place == 'USNew York'][['elapsed'] + [c for c in data.columns if 'ses_4d_prior_slope' in c]]
# In[237]:


slope_cols[:5]

data
# In[238]:


for col in [c for c in data.columns if any(z in c for z in
                                           ['vs_continent', 'nearest', 'vs_world', 'surrounding_places'])]:
    #     print(col)
    data[col + '_times_days'] = data[col] * data.elapsed

# In[239]:


data['saturation_slope_ConfirmedCases'] = (- np.log(data.ConfirmedCases + 1) + np.log(
    data.TRUE_POPULATION + 1)) / data.elapsed
data['saturation_slope_Fatalities'] = (- np.log(data.Fatalities + 1) + np.log(data.TRUE_POPULATION + 1)) / data.elapsed

data['dist_to_ConfirmedCases_saturation_times_days'] = (- np.log(data.ConfirmedCases + 1) + np.log(
    data.TRUE_POPULATION + 1)) * data.elapsed
data['dist_to_Fatalities_saturation_times_days'] = (- np.log(data.Fatalities + 1) + np.log(
    data.TRUE_POPULATION + 1)) * data.elapsed

data['slope_to_1pct_ConfirmedCases'] = (- np.log(data.ConfirmedCases + 1) + np.log(
    data.TRUE_POPULATION + 1) - 4.6) / data.elapsed
data['slope_to_0.1pct_Fatalities'] = (- np.log(data.Fatalities + 1) + np.log(
    data.TRUE_POPULATION + 1) - 6.9) / data.elapsed

data['dist_to_1pct_ConfirmedCases_times_days'] = (- np.log(data.ConfirmedCases + 1) + np.log(
    data.TRUE_POPULATION + 1) - 4.6) * data.elapsed
data['dist_to_0.1pct_Fatalities_times_days'] = (- np.log(data.Fatalities + 1) + np.log(
    data.TRUE_POPULATION + 1) - 6.9) * data.elapsed

data.ConfirmedCases_12d_prior_slope.plot(kind='hist')
# In[240]:


data['trendline_per_capita_ConfirmedCases_4d_slope'] = (np.log(data.ConfirmedCases + 1) - np.log(
    data.TRUE_POPULATION + 1)) + (data.ConfirmedCases_4d_prior_slope * data.elapsed)
data['trendline_per_capita_ConfirmedCases_7d_slope'] = (np.log(data.ConfirmedCases + 1) - np.log(
    data.TRUE_POPULATION + 1)) + (data.ConfirmedCases_7d_prior_slope * data.elapsed)

data['trendline_per_capita_Fatalities_12d_slope'] = (np.log(data.Fatalities + 1) - np.log(data.TRUE_POPULATION + 1)) + (
            data.ConfirmedCases_12d_prior_slope * data.elapsed)
data['trendline_per_capita_Fatalities_20d_slope'] = (np.log(data.Fatalities + 1) - np.log(data.TRUE_POPULATION + 1)) + (
            data.ConfirmedCases_20d_prior_slope * data.elapsed)

# In[ ]:


data[data.Place == 'USNew York']
# In[241]:


len(data)

data.CaseChgRate.plot(kind='hist', bins=250);
# In[ ]:


data_bk = data.copy()
# In[ ]:


# In[242]:


data.groupby('Place').last()

# In[ ]:


# In[ ]:


# data['log_days_since_first_case'] =  np.log(data.days_since_first_case + 1)
# data['log_days_since_first_fatality'] = np.log(data.days_since_first_fatality + 1)

data['sqrt_days_since_first_case'] = np.sqrt(data.days_since_first_case)
data['sqrt_days_since_first_fatality'] = np.sqrt(data.days_since_first_fatality)

# In[ ]:


# In[243]:


data.loc[4368]


# #### Drop Anyone with Only 1 Case at Init

# In[244]:


def logHist(x, b=150):
    np.log(x + 1).plot(kind='hist', bins=b)


logHist(data.ConfirmedCases)
# In[ ]:


# In[ ]:


# In[ ]:


# #### And Drop Anyone Who's Moved to Flat Prior to Date_i
len(data)
data[(data.ConfirmedCases_7d_prior_slope < .01) & (data.ConfirmedCases > 20)
     & (data.days_since_first_case > 20)
     ].groupby('Country').count()
len(data[data.ConfirmedCases > 1])
# In[ ]:


data = data.drop(index=data[(data.ConfirmedCases_7d_prior_slope < .01) & (data.ConfirmedCases > 20)
                            & (data.days_since_first_case > 20)
                            ].index)
# In[245]:


logHist(data.ConfirmedCases_4d_prior_slope, 50)

# In[246]:


logHist(data.ConfirmedCases_7d_prior_slope, 50)

np.std(x.log_cases)
np.std(x.log_fatalities)
# In[ ]:


# In[247]:


data['log_fatalities'] = np.log(data.Fatalities + 1)  # + 0.4 * np.random.normal(0, 1, len(data))
data['log_cases'] = np.log(data.ConfirmedCases + 1)  # + 0.2 *np.random.normal(0, 1, len(data))

data.log_cases.plot(kind='hist', bins=250)
# In[248]:


data['is_China'] = (data.Country == 'China') & (~data.Place.isin(['Hong Kong', 'Macau']))

# In[249]:


for col in [c for c in data.columns if 'd_ewm' in c]:
    data[col] += np.random.normal(0, 1, len(data)) * np.std(data[col]) * 0.2

data[data.log_cfr > -11].log_fatalities.plot(kind='hist', bins=150)
# In[250]:


data['is_province'] = 1.0 * (~data.Province_State.isnull())

# In[251]:


data['log_elapsed'] = np.log(data.elapsed + 1)

# In[252]:


data.columns

# In[253]:


data.columns[::19]

# In[254]:


data.shape

# In[255]:


logHist(data.ConfirmedCases)

# In[ ]:


# In[ ]:


# ### Data Cleanup

# In[256]:


data.drop(columns=['TRUE_POPULATION'], inplace=True)

# In[257]:


data['final_day_of_week'] = data.Date_f.apply(datetime.datetime.weekday)

# In[258]:


data['base_date_day_of_week'] = data.Date.apply(datetime.datetime.weekday)

# In[259]:


data['date_difference_modulo_7_days'] = (data.Date_f - data.Date).dt.days % 7

for c in data.columns.to_list():
    if 'days_since' in c:
        data[c] = np.log(data[c] + 1)
# In[ ]:


# In[260]:


for c in data.columns.to_list():
    if 'days_to' in c:
        #         print(c)
        data[c] = data[c].where(~np.isinf(data[c]), 1e3)
        data[c] = np.clip(data[c], 0, 365)
        data[c] = np.sqrt(data[c])

# In[ ]:


# ## II. Modeling

# ### Data Prep

# In[261]:


model_data = data[((len(test) == 0) | (data.Date_f < test.Date.min()))
                  &
                  (data.ConfirmedCases > 0) &
                  (~data.ConfirmedCases_f.isnull())].copy()

data.Date_f
# In[262]:


test.Date.min()

# In[263]:


model_data.Date_f.max()

# In[264]:


model_data.Date_f.max()

# In[265]:


model_data.Date.max()

# In[266]:


model_data.Date_f.min()

# In[ ]:


# In[267]:


model_data = model_data[~(
        (np.random.rand(len(model_data)) < 0.8) &
        (model_data.Country == 'China') &
        (model_data.Date < datetime.datetime(2020, 2, 15)))]

# In[268]:


x_dates = model_data[['Date_i', 'Date_f', 'Place']]

# In[269]:


x = model_data[
    model_data.columns.to_list()[
    model_data.columns.to_list().index('ConfirmedCases_1d_prior_value'):]] \
    .drop(columns=['Date_i', 'Date_f', 'CaseChgRate', 'FatalityChgRate'])


# #### Data Obfuscation and Smoothing/Blending to Avoid Overfit
for c in data.columns.to_list():
    if 'trendline_per' in c:
        print(c)
        h = x[c].plot(kind='hist', bins=100)
        h = plt.figure()
        print(np.std(x[c]))[(c, np.min(x[c]), np.max(x[c]))
        for c in x.columns.to_list():
            if np.max(x[c]) - np.min(x[c]) > 20]:
                for c in x.columns.to_list()[20:25]:
                    print(c)
                    pf = x[c].plot(kind='hist', bins=250);
        pf = plt.figure()
        x_full = data[x.columns].copy()
        x_full.shape

    # ### Cluster the Test Set

    test.Date.max()

    if PRIVATE:
        data_test = data[(data.Date_i == train.Date.max()) &
                         (data.Date_f.isin(test.Date.unique()))].copy()
    else:
        data_test = data[(data.Date_i == test.Date.min() - datetime.timedelta(1)) &
                         (data.Date_f.isin(test.Date.unique()))].copy()


    data_test.Date.unique()
    test.Date.unique()

    x_test = data_test[x.columns].copy()


    train.Date.max()

    test.Date.max()

    data_test[data_test.Place == 'San Marino'].Date_fdata_test.groupby('Place').Date_f.count().sort_values()


    if MODEL_Y is 'slope':
        y_cases = model_data.CaseChgRate
        y_fatalities = model_data.FatalityChgRate
    else:
        y_cases = model_data.CaseChgRate * model_data.elapsed
        y_fatalities = model_data.FatalityChgRate * model_data.elapsed

    y_cfr = np.log((model_data.Fatalities_f + np.clip(0.015 * model_data.ConfirmedCases_f, 0, 0.3)) / (
                model_data.ConfirmedCases_f + 0.1))

    # In[282]:

    groups = model_data.Country
    places = model_data.Place

    # #### Model Setup

    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit, PredefinedSplit
    from sklearn.model_selection import ParameterSampler
    from sklearn.metrics import make_scorer
    from sklearn.ensemble import ExtraTreesRegressor
    from xgboost import XGBRegressor
    from sklearn.linear_model import HuberRegressor, ElasticNet
    import lightgbm as lgb

    enet_params = {'alpha': [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, ],
                   'l1_ratio': [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.97, 0.99]}

    et_params = {'n_estimators': [50, 70, 100, 140],
                 'max_depth': [3, 5, 7, 8, 9, 10],
                 'min_samples_leaf': [30, 50, 70, 100, 130, 165, 200, 300, 600],
                 'max_features': [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
                 'min_impurity_decrease': [0, 1e-5],  # 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                 'bootstrap': [True, False],  # False is clearly worse
                 #   'criterion': ['mae'],
                 }

    # In[286]:

    lgb_params = {
        'max_depth': [5, 12],
        'n_estimators': [100, 200, 300, 500],  # continuous
        'min_split_gain': [0, 0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
        'min_child_samples': [10, 20, 30, 40, 70, 100, 170, 300, 450, 700, 1000, 2000],
        'min_child_weight': [0],  # , 1e-3],
        'num_leaves': [5, 10, 20, 30],
        'learning_rate': [0.05, 0.07, 0.1],  # , 0.1],
        'colsample_bytree': [0.1, 0.2, 0.33, 0.5, 0.65, 0.8, 0.9],
        'colsample_bynode': [0.1, 0.2, 0.33, 0.5, 0.65, 0.81],
        'reg_lambda': [1e-5, 3e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, ],
        'reg_alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 30, 1000, ],  # 1, 10, 100, 1000, 10000],
        'subsample': [0.8, 0.9, 1],
        'subsample_freq': [1],
        'max_bin': [7, 15, 31, 63, 127, 255],
        'extra_trees': [True, False],
        #                 'boosting': ['gbdt', 'dart'],
        #     'subsample_for_bin': [200000, 500000],
    }

    # In[287]:

    MSE = 'neg_mean_squared_error'
    MAE = 'neg_mean_absolute_error'


    # In[ ]:

    # In[288]:

    def trainENet(x, y, groups, cv=0, **kwargs):
        return trainModel(x, y, groups,
                          clf=ElasticNet(normalize=True, selection='random',
                                         max_iter=3000),
                          params=enet_params,
                          cv=cv, **kwargs)


    # In[289]:

    def trainETR(x, y, groups, cv=0, n_jobs=5, **kwargs):
        clf = ExtraTreesRegressor(n_jobs=1)
        params = et_params
        return trainModel(x, y, groups, clf, params, cv, n_jobs, **kwargs)


    # In[290]:

    def trainLGB(x, y, groups, cv=0, n_jobs=4, **kwargs):
        clf = lgb.LGBMRegressor(verbosity=0, hist_pool_size=1000,
                                )
        params = lgb_params

        return trainModel(x, y, groups, clf, params, cv, n_jobs, **kwargs)


    # In[291]:

    def trainModel(x, y, groups, clf, params, cv=0, n_jobs=None,
                   verbose=0, splits=None, **kwargs):
        #     if cv is 0:
        #         param_sets = list(ParameterSampler(params, n_iter=1))
        #         clf = clf.set_params(**param_sets[0] )
        #         if n_jobs is not None:
        #             clf = clf.set_params(** {'n_jobs': n_jobs } )
        #         f = clf.fit(x, y)
        #         return clf
        #     else:
        if n_jobs is None:
            n_jobs = 4
        if np.random.rand() < 0.8:  # all shuffle, don't want overfit models, just reasonable
            folds = GroupShuffleSplit(n_splits=4,
                                      test_size=0.2 + 0.10 * np.random.rand())
        else:
            folds = GroupKFold(4)
        clf = RandomizedSearchCV(clf, params,
                                 cv=folds,
                                 #                                  cv = GroupKFold(4),
                                 n_iter=10,
                                 verbose=0, n_jobs=n_jobs, scoring=MSE)
        f = clf.fit(x, y, groups)
        if verbose > 0:
            print(pd.DataFrame(clf.cv_results_));
            print();
            #   print(pd.DataFrame(clf.cv_results_).to_string()); print();

        best = clf.best_estimator_;
        print(best)
        print("Best Score: {}".format(np.round(clf.best_score_, 4)))

        return best


    # In[292]:

    np.mean(y_cases)


    # In[ ]:

    # In[293]:

    def getSparseColumns(x, verbose=0):
        sc = []
        for c in x.columns.to_list():
            u = len(x[c].unique())
            if u > 10 and u < 0.01 * len(x):
                sc.append(c)
                if verbose > 0:
                    print("{}: {}".format(c, u))

        return sc

    def noisify(x, noise=0.1):
        x = x.copy()
        # cols = x.columns.to_list()
        cols = getSparseColumns(x)
        for c in cols:
            u = len(x[c].unique())
            if u > 50:
                x[c].values[:] = x[c].values + np.random.normal(0, noise, len(x)) * np.std(x[c])

        return x




def getMaxOverlap(row, df):
    #     max_overlap_frac = 0

    df_place = df[df.Place == row.Place]
    if len(df_place) == 0:
        return 0
    #     print(df_place)
    overlap = (np.clip(df_place.Date_f, None, row.Date_f) - np.clip(df_place.Date_i, row.Date_i, None)).dt.days
    overlap = np.clip(overlap, 0, None)
    length = np.clip((df_place.Date_f - df_place.Date_i).dt.days,
                     (row.Date_f - row.Date_i).days, None)

    return np.amax(overlap / length)


def getSampleWeight(x, groups):
    counter = Counter(groups)
    median_count = np.median([counter[group] for group in groups.unique()])
    #     print(median_count)
    c_count = [counter[group] for group in groups]

    e_decay = np.round(LT_DECAY_MIN + np.random.rand() * (LT_DECAY_MAX - LT_DECAY_MIN), 1)
    print("LT weight decay: {:.2f}".format(e_decay));
    ssr = np.power(1 / np.clip(c_count / median_count, 0.1, 30),
                   0.1 + np.random.rand() * 0.6) \
          / np.power(x.elapsed / 3, e_decay) \
          * SET_FRAC * np.exp(-    np.random.rand())

    #     print(np.power(  1 / np.clip( c_count / median_count , 1,  10) ,
    #                         0.1 + np.random.rand() * 0.3))
    #     print(np.power(x.elapsed / 3, e_decay))
    #     print(np.exp(  1.5 * (np.random.rand() - 0.5) ))

    # drop % of groups at random
    group_drop = dict([(group, np.random.rand() < 0.15) for group in groups.unique()])
    ssr = ssr * ([1 - group_drop[group] for group in groups])
    #     print(ssr[::171])
    #     print(np.array([ 1 -group_drop[group] for group in groups]).sum() / len(groups))

    #     pd.Series(ssr).plot(kind='hist', bins = 100)
    return ssr;


group_drop = dict([(group, np.random.rand() < 0.20) for group in groups.unique()])

np.array([1 - group_drop[group] for group in groups]).sum() / len(groups)
[c for c in x.columns if 'continent' in c]
x.columns[::10]
x.shapecontain_data.columns


# In[297]:


def runBags(x, y, groups, cv, bags=3, model_type=trainLGB,
            noise=0.1, splits=None, weights=None, **kwargs):
    models = []
    for bag in range(bags):
        print("\nBAG {}".format(bag + 1))

        x = x.copy()  # copy X to modify it with noise

        if DROPS:
            # drop 0-70% of the bend/slope/prior features, just for speed and model diversity
            for col in [c for c in x.columns if any(z in c for z in ['bend', 'slope', 'prior'])]:
                if np.random.rand() < np.sqrt(np.random.rand()) * 0.7:
                    x[col].values[:] = 0

        # 00% of the time drop all 'rate_since' features
        #         if np.random.rand() < 0.00:
        #             print('dropping rate_since features')
        #             for col in [c for c in x.columns if 'rate_since' in c]:
        #                 x[col].values[:] = 0

        # 20% of the time drop all 'world' features
        #         if np.random.rand() < 0.00:
        #             print('dropping world features')
        #             for col in [c for c in x.columns if 'world' in c]:
        #                 x[col].values[:] = 0

        # % of the time drop all 'nearest' features
        if DROPS and (np.random.rand() < 0.30):
            print('dropping nearest features')
            for col in [c for c in x.columns if 'nearest' in c]:
                x[col].values[:] = 0

        #  % of the time drop all 'surrounding_places' features
        if DROPS and (np.random.rand() < 0.25):
            print('dropping \'surrounding places\' features')
            for col in [c for c in x.columns if 'surrounding_places' in c]:
                x[col].values[:] = 0

        # 20% of the time drop all 'continent' features
        #         if np.random.rand() < 0.20:
        #             print('dropping continent features')
        #             for col in [c for c in x.columns if 'continent' in c]:
        #                 x[col].values[:] = 0

        # drop 0-50% of all features
        #         if DROPS:
        col_drop_frac = np.sqrt(np.random.rand()) * 0.5
        for col in [c for c in x.columns if 'elapsed' not in c]:
            if np.random.rand() < col_drop_frac:
                x[col].values[:] = 0

        x = noisify(x, noise)

        if DROPS and (np.random.rand() < SUP_DROP):
            print("Dropping supplemental country data")
            for col in x[[c for c in x.columns if c in sup_data.columns]]:
                x[col].values[:] = 0

        if DROPS and (np.random.rand() < ACTIONS_DROP):
            for col in x[[c for c in x.columns if c in contain_data.columns]]:
                x[col].values[:] = 0
        #             print(x.StringencyIndex_20d_ewm[::157])
        else:
            print("*using containment data")

        if np.random.rand() < 0.6:
            x.S_data_days = 0

        ssr = getSampleWeight(x, groups)

        date_falloff = 0 + (1 / 30) * np.random.rand()
        if weights is not None:
            ssr = ssr * np.exp(-weights * date_falloff)

        ss = (np.random.rand(len(y)) < ssr)
        print("n={}".format(len(x[ss])))

        p1 = x.elapsed[ss].plot(kind='hist', bins=int(x.elapsed.max() - x.elapsed.min() + 1))
        p1 = plt.figure();
        #         break
        print(Counter(groups[ss]))
        print((ss).sum())
        models.append(model_type(x[ss], y[ss], groups[ss], cv, **kwargs))
    return models


# In[298]:


x = x.astype(np.float32)

x.elapsed
# In[ ]:


# In[299]:


BAG_MULT = 1

# In[ ]:


# In[300]:


x.shape

# In[ ]:


# In[301]:


lgb_c_clfs = [];
lgb_c_noise = []

# In[302]:


date_weights = np.abs((model_data.Date_f - test.Date.min()).dt.days)

# In[303]:


for iteration in range(0, int(math.ceil(BAGS * 1.1))):
    for noise in [0.05, 0.1, 0.2, 0.3, 0.4]:
        print("\n---\n\nNoise of {}".format(noise));
        num_bags = 1 * BAG_MULT;
        if np.random.rand() < PLACE_FRACTION:
            cv_group = places
            print("CV by Place")
        else:
            cv_group = groups
            print("CV by Country")

        lgb_c_clfs.extend(runBags(x, y_cases,
                                  cv_group,  # groups
                                  MSE, num_bags, trainLGB, verbose=0,
                                  noise=noise, weights=date_weights

                                  ))
        lgb_c_noise.extend([noise] * num_bags)
        if SINGLE_MODEL:
            break;

# In[ ]:


np.isinf(x).sum().sort_values()
# In[ ]:


enet_c_clfs = runBags(x, y_cases, groups, MSE, 1, trainENet, verbose=1)
# In[ ]:


# In[304]:


lgb_f_clfs = [];
lgb_f_noise = []

# In[305]:


for iteration in range(0, int(np.ceil(np.sqrt(BAGS)))):
    for noise in [0.5, 1, 2, 3, ]:
        print("\n---\n\nNoise of {}".format(noise));
        num_bags = 1 * int(np.ceil(np.sqrt(BAG_MULT)))
        if np.random.rand() < PLACE_FRACTION:
            cv_group = places
            print("CV by Place")
        else:
            cv_group = groups
            print("CV by Country")

        lgb_f_clfs.extend(runBags(x, y_fatalities,
                                  cv_group,  # places, # groups,
                                  MSE, num_bags, trainLGB,
                                  verbose=0, noise=noise,
                                  weights=date_weights
                                  ))
        lgb_f_noise.extend([noise] * num_bags)
        if SINGLE_MODEL:
            break;

lgb_f_noise = lgb_f_noise[0:3]
lgb_f_clfs = lgb_f_clfs[0:3]
lgb_f_noise = lgb_f_noise[2:]
lgb_f_clfs = lgb_f_clfs[2:]
et_f_clfs = runBags(x, y_fatalities, groups, MSE, 1, trainETR, verbose=1)

enet_f_clfs = runBags(x, y_fatalities, groups, MSE, 1, trainENet, verbose=1)

y_cfr.plot(kind='hist', bins=250)
# In[306]:


lgb_cfr_clfs = [];
lgb_cfr_noise = [];

# In[307]:


for iteration in range(0, int(np.ceil(np.sqrt(BAGS)))):
    for noise in [0.4, 1, 2, 3]:
        print("\n---\n\nNoise of {}".format(noise));
        num_bags = 1 * BAG_MULT;
        if np.random.rand() < 0.5 * PLACE_FRACTION:
            cv_group = places
            print("CV by Place")
        else:
            cv_group = groups
            print("CV by Country")

        lgb_cfr_clfs.extend(runBags(x, y_cfr,
                                    cv_group,  # groups
                                    MSE, num_bags, trainLGB, verbose=0,
                                    noise=noise,
                                    weights=date_weights

                                    ))
        lgb_cfr_noise.extend([noise] * num_bags)
        if SINGLE_MODEL:
            break;

x_test
# In[308]:


lgb_cfr_clfs[0].predict(x_test)


# In[309]:


# full sample, through 03/28 (avail on 3/30), lgb only: 0.0097 / 0.0036;   0.0092 / 0.0042
#


# ##### Feature Importance

# In[310]:


def show_FI(model, featNames, featCount):
    # show_FI_plot(model.feature_importances_, featNames, featCount)
    fis = model.feature_importances_
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(fis)[::-1][:featCount]
    g = sns.barplot(y=featNames[indices][:featCount],
                    x=fis[indices][:featCount], orient='h')
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title(" feature importance")


# In[311]:


def avg_FI(all_clfs, featNames, featCount):
    # 1. Sum
    clfs = []
    for clf_set in all_clfs:
        for clf in clf_set:
            clfs.append(clf);
    print("{} classifiers".format(len(clfs)))
    fi = np.zeros((len(clfs), len(clfs[0].feature_importances_)))
    for idx, clf in enumerate(clfs):
        fi[idx, :] = clf.feature_importances_
    avg_fi = np.mean(fi, axis=0)

    # 2. Plot
    fis = avg_fi
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(fis)[::-1]  # [:featCount]
    # print(indices)
    g = sns.barplot(y=featNames[indices][:featCount],
                    x=fis[indices][:featCount], orient='h')
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title(" feature importance")

    return pd.Series(fis[indices], featNames[indices])


# In[312]:


def linear_FI_plot(fi, featNames, featCount):
    # show_FI_plot(model.feature_importances_, featNames, featCount)
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(np.absolute(fi))[::-1]  # [:featCount]
    g = sns.barplot(y=featNames[indices][:featCount],
                    x=fi[indices][:featCount], orient='h')
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title(" feature importance")
    return pd.Series(fi[indices], featNames[indices])


# In[ ]:


fi_list = []
for clf in enet_c_clfs:
    fi = clf.coef_ * np.std(x, axis=0).values
    fi_list.append(fi)
fis = np.mean(np.array(fi_list), axis=0)
fis = linear_FI_plot(fis, x.columns.values, 25)
lgb_c_clfs
# In[313]:


f = avg_FI([lgb_c_clfs], x.columns, 25)

# In[314]:


for feat in ['bend', 'capita', 'cfr', 'slope', 'since', 'chg', 'ersonal',
             'world', 'continent', 'nearest', 'surrounding']:
    print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))

# In[ ]:


# In[315]:


f = avg_FI([lgb_f_clfs], x.columns, 25)

# In[316]:


for feat in ['bend', 'capita', 'cfr', 'slope', 'since', 'chg', 'ersonal',
             'world', 'continent', 'nearest', 'surrounding']:
    print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))

# In[ ]:


x.days_since_Stringency_1.plot(kind='hist', bins=100)
len(x.log_fatalities.unique())
# In[317]:


f = avg_FI([lgb_cfr_clfs], x.columns, 25)

# In[318]:


for feat in ['bend', 'capita', 'cfr', 'slope', 'since', 'chg', 'ersonal',
             'world', 'continent', 'nearest', 'surrounding']:
    print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))



np.corrcoef((x.S_data_days.fillna(0), y_cases, y_fatalities, y_cfr))
np.corrcoef((x.murder.fillna(0), y_cases, y_fatalities, y_cfr))
np.corrcoef((x.Personality_uai, y_cases, y_fatalities, y_cfr))
np.corrcoef((x.TFR, y_cases, y_fatalities, y_cfr))
np.corrcoef((x.Avg_age, y_cases, y_fatalities, y_cfr))
np.corrcoef((x.High_rises, y_cases, y_fatalities, y_cfr))
np.corrcoef((x.max_high_rises, y_cases, y_fatalities, y_cfr))
data.elapsedfi_list = []
for clf in enet_f_clfs:
    fi = clf.coef_ * np.std(x, axis=0).values
    fi_list.append(fi)
fis = np.mean(np.array(fi_list), axis=0)
fis = linear_FI_plot(fis, x.columns.values, 25)
# In[ ]:


# In[319]:


all_c_clfs = [lgb_c_clfs, ]  # enet_c_clfs]
all_f_clfs = [lgb_f_clfs]  # , enet_f_clfs]
all_cfr_clfs = [lgb_cfr_clfs]

# In[320]:


all_c_noise = [lgb_c_noise]
all_f_noise = [lgb_f_noise]
all_cfr_noise = [lgb_cfr_noise]

# In[ ]:


# In[321]:


NUM_TEST_RUNS = 1

# In[322]:


c_preds = np.zeros((NUM_TEST_RUNS * sum([len(x) for x in all_c_clfs]), len(x_test)))
f_preds = np.zeros((NUM_TEST_RUNS * sum([len(x) for x in all_f_clfs]), len(x_test)))
cfr_preds = np.zeros((NUM_TEST_RUNS * sum([len(x) for x in all_cfr_clfs]), len(x_test)))


# In[323]:


def avg(x):
    return (np.mean(x, axis=0) + np.median(x, axis=0)) / 2


# In[324]:


count = 0

for idx, clf in enumerate(lgb_c_clfs):
    for i in range(0, NUM_TEST_RUNS):
        noise = lgb_c_noise[idx]
        c_preds[count, :] = np.clip(clf.predict(noisify(x_test, noise)), -1, 10)
        count += 1
# y_cases_pred_blended_full = avg(c_preds)


# In[325]:


count = 0

for idx, clf in enumerate(lgb_f_clfs):
    for i in range(0, NUM_TEST_RUNS):
        noise = lgb_f_noise[idx]
        f_preds[count, :] = np.clip(clf.predict(noisify(x_test, noise)), -1, 10)
        count += 1
# y_fatalities_pred_blended_full = avg(f_preds)


# In[326]:


count = 0

for idx, clf in enumerate(lgb_cfr_clfs):
    for i in range(0, NUM_TEST_RUNS):
        noise = lgb_cfr_noise[idx]
        cfr_preds[count, :] = np.clip(clf.predict(noisify(x_test, noise)), -10, 10)
        count += 1


# y_cfr_pred_blended_full = avg(cfr_preds)


# In[ ]:


# In[ ]:


# In[327]:


def qPred(preds, pctile, simple=False):
    q = np.percentile(preds, pctile, axis=0)
    if simple:
        return q;
    resid = preds - q
    resid_wtg = 2 / 100 / len(preds) * (np.clip(resid, 0, None) * (pctile) + np.clip(resid, None, 0) * (100 - pctile))
    adj = np.sum(resid_wtg, axis=0)
    #     print(q)
    #     print(adj)
    #     print(q+adj)
    return q + adj


# In[ ]:


# In[328]:


q = 50

# In[329]:


y_cases_pred_blended_full = qPred(c_preds, q)  # avg(c_preds)
y_fatalities_pred_blended_full = qPred(f_preds, q)  # avg(f_preds)
y_cfr_pred_blended_full = qPred(cfr_preds, q)  # avg(cfr_preds)

# In[ ]:


cfr_predslgb_cfr_noiselgb_cfr_clfs[0].predict(noisify(x_test, 0.4))
cfr_preds[0][0:500]
x.log_cfr.plot(kind='hist', bins=250)
# In[ ]:


# In[330]:


np.mean(np.corrcoef(c_preds[::NUM_TEST_RUNS]), axis=0)

# In[331]:


np.mean(np.corrcoef(f_preds[::NUM_TEST_RUNS]), axis=0)

# In[332]:


np.mean(np.corrcoef(cfr_preds[::NUM_TEST_RUNS]), axis=0)

cfr_preds
# In[333]:


pd.Series(np.std(c_preds, axis=0)).plot(kind='hist', bins=50)

# In[334]:


pd.Series(np.std(f_preds, axis=0)).plot(kind='hist', bins=50)

# In[335]:


pd.Series(np.std(cfr_preds, axis=0)).plot(kind='hist', bins=50)

# In[336]:


y_cfr

# In[337]:


(groups == 'Sierra Leone').sum()

# In[338]:


pred = pd.DataFrame(np.hstack((np.transpose(c_preds),
                               np.transpose(f_preds))), index=x_test.index)
pred['Place'] = data_test.Place

pred['Date'] = data_test.Date
pred['Date_f'] = data_test.Date_f

# In[339]:


pred[(pred.Date == pred.Date.max()) & (pred.Date_f == pred.Date_f.max())][30: 60]

# In[340]:


(pred.Place == 'Sierra Leone').sum()

# In[341]:


np.round(pred[(pred.Date == pred.Date.max()) & (pred.Date_f == pred.Date_f.max())], 2)[190:220:]

# In[342]:


np.round(pred[(pred.Date == pred.Date.max()) & (pred.Date_f == pred.Date_f.max())][220:-20], 2)

# In[343]:


c_preds.shape
x_test.shape

data_test.shapepd.DataFrame({'c_mean': np.mean(c_preds, axis=0),
                             'c_median': np.median(c_preds, axis=0),
                             }, index=data_test.Place)[::7]
np.median(c_preds, axis=0)[::71]
# In[ ]:


# ### III. Other

# In[ ]:


MAX_DATE = np.max(train.Date)
final = train[train.Date == MAX_DATE]
# In[ ]:


# In[ ]:


train.groupby('Place')[['ConfirmedCases', 'Fatalities']].apply(lambda x: np.sum(x > 0))
num_changes = train.groupby('Place')[['ConfirmedCases', 'Fatalities']].apply(lambda x: np.sum(x - x.shift(1) > 0))
num_changes.Fatalities.plot(kind='hist', bins=50);
num_changes.ConfirmedCases.plot(kind='hist', bins=50);


# In[ ]:


# In[ ]:


# ### Rate Calculation
def getRate(train, window=5):
    joined = pd.merge(train[train.Date ==
                            np.max(train.Date) - datetime.timedelta(window)],
                      final, on=['Place'])
    joined['FatalityRate'] = (np.log(joined.Fatalities_y + 1) \
                              - np.log(joined.Fatalities_x + 1)) / window
    joined['CasesRate'] = (np.log(joined.ConfirmedCases_y + 1) \
                           - np.log(joined.ConfirmedCases_x + 1)) / window
    joined.set_index('Place', inplace=True)

    rates = joined[[c for c in joined.columns.to_list() if 'Rate' in c]]
    return ratesltr = getRate(train, 14)
    lm = pd.merge(ltr, num_changes, on='Place')
    lm.filter(like='China', axis='rows')
    flat = lm[
        (lm.CasesRate < 0.01) & (lm.ConfirmedCases > 5)]
    flatc_rate = pd.Series(
        np.where(num_changes.ConfirmedCases >= 0,
                 getRate(train, 7).CasesRate,
                 getRate(train, 5).CasesRate),
        index=num_changes.index, name='CasesRate')


f_rate = pd.Series(
    np.where(num_changes.Fatalities >= 0,
             getRate(train, 7).FatalityRate,
             getRate(train, 4).CasesRate),
    index=num_changes.index, name='FatalityRate')


# In[ ]:


# ### Plot of Changes
def rollDates(df, i):
    df = df.copy()
    df.Date = df.Date + datetime.timedelta(i)
    return dfm = pd.merge(rollDates(train, 7), train, on=['Place', 'Date'])


m['CaseChange'] = (np.log(m.ConfirmedCases_y + 1) - np.log(m.ConfirmedCases_x + 1)) / 7
m[m.Place == 'USMaine']
# In[ ]:


# #### Histograms of Case Counts

# In[ ]:


m = pd.merge(rollDates(full_train, 1), full_train, on=['Place', 'Date'])

# In[ ]:


# ##### CFR Charts
joined.Fatalities_ywithcases = joined[joined.ConfirmedCases_y > 300]
withcases.sort_values(by=['Fatalities_y'])(withcases.Fatalities_y / withcases.ConfirmedCases_x).plot(kind='hist',
                                                                                                     bins=150);
(final.Fatalities / final.ConfirmedCases).plot(kind='hist', bins=250);
# In[ ]:


# In[ ]:


# ### Predict on Test Set

# In[344]:


data_wp = data_test.copy()

# In[345]:


if MODEL_Y is 'slope':
    data_wp['case_slope'] = y_cases_pred_blended_full
    data_wp['fatality_slope'] = y_fatalities_pred_blended_full
else:
    data_wp['case_slope'] = y_cases_pred_blended_full / x_test.elapsed
    data_wp['fatality_slope'] = y_fatalities_pred_blended_full / x_test.elapsed

data_wp['cfr_pred'] = y_cfr_pred_blended_full

# In[346]:


data_wp.head()

# In[347]:


data_wp.shape

# In[348]:


data_wp.Date_f.unique()

# In[349]:


train.Date.max()

data_wp.Date
# In[350]:


test.Date.min()

test
# In[351]:


if len(test) > 0:
    base_date = test.Date.min() - datetime.timedelta(1)
else:
    base_date = train.Date.max()

train
# In[352]:


len(test)

# In[353]:


base_date

# In[354]:


data_wp_ss = data_wp[data_wp.Date == base_date]
data_wp_ss = data_wp_ss.drop(columns='Date').rename(columns={'Date_f': 'Date'})

base_date
# In[355]:


data_wp_ss.head()

# In[356]:


test

data_wp_ss.columns
# In[ ]:


len(test);
len(x_test)
# In[357]:


test_wp = pd.merge(test, data_wp_ss[['Date', 'Place', 'case_slope', 'fatality_slope', 'cfr_pred',
                                     'elapsed']],
                   how='left', on=['Date', 'Place'])

test_wp[test_wp.Country == 'US']
test_wp
# In[358]:


first_c_slope = test_wp[~test_wp.case_slope.isnull()].groupby('Place').first()
last_c_slope = test_wp[~test_wp.case_slope.isnull()].groupby('Place').last()

first_f_slope = test_wp[~test_wp.fatality_slope.isnull()].groupby('Place').first()
last_f_slope = test_wp[~test_wp.fatality_slope.isnull()].groupby('Place').last()

first_cfr_pred = test_wp[~test_wp.cfr_pred.isnull()].groupby('Place').first()
last_cfr_pred = test_wp[~test_wp.cfr_pred.isnull()].groupby('Place').last()

test_wpfirst_c_slope
# In[359]:


test_wp.describe()

# In[360]:


test_wp

# In[361]:


test_wp.case_slope = np.where(test_wp.case_slope.isnull() &
                              (test_wp.Date < first_c_slope.loc[test_wp.Place].Date.values),

                              first_c_slope.loc[test_wp.Place].case_slope.values,
                              test_wp.case_slope
                              )

test_wp.case_slope = np.where(test_wp.case_slope.isnull() &
                              (test_wp.Date > last_c_slope.loc[test_wp.Place].Date.values),

                              last_c_slope.loc[test_wp.Place].case_slope.values,
                              test_wp.case_slope
                              )

# In[362]:


test_wp.fatality_slope = np.where(test_wp.fatality_slope.isnull() &
                                  (test_wp.Date < first_f_slope.loc[test_wp.Place].Date.values),

                                  first_f_slope.loc[test_wp.Place].fatality_slope.values,
                                  test_wp.fatality_slope
                                  )

test_wp.fatality_slope = np.where(test_wp.fatality_slope.isnull() &
                                  (test_wp.Date > last_f_slope.loc[test_wp.Place].Date.values),

                                  last_f_slope.loc[test_wp.Place].fatality_slope.values,
                                  test_wp.fatality_slope
                                  )

# In[363]:


test_wp.cfr_pred = np.where(test_wp.cfr_pred.isnull() &
                            (test_wp.Date < first_cfr_pred.loc[test_wp.Place].Date.values),

                            first_cfr_pred.loc[test_wp.Place].cfr_pred.values,
                            test_wp.cfr_pred
                            )

test_wp.cfr_pred = np.where(test_wp.cfr_pred.isnull() &
                            (test_wp.Date > last_cfr_pred.loc[test_wp.Place].Date.values),

                            last_cfr_pred.loc[test_wp.Place].cfr_pred.values,
                            test_wp.cfr_pred
                            )

# In[ ]:


# In[364]:


test_wp.case_slope = test_wp.case_slope.interpolate('linear')
test_wp.fatality_slope = test_wp.fatality_slope.interpolate('linear')
test_wp.cfr_pred = test_wp.cfr_pred.interpolate('linear')

# In[365]:


test_wp.case_slope = test_wp.case_slope.fillna(0)
test_wp.fatality_slope = test_wp.fatality_slope.fillna(0)

# test_wp.fatality_slope = test_wp.fatality_slope.fillna(0)

test_wp.cfr_pred.isnull().sum()
# #### Convert Slopes to Aggregate Counts

# In[366]:


LAST_DATE = test.Date.min() - datetime.timedelta(1)

# In[367]:


final = train_bk[train_bk.Date == LAST_DATE]

trainfinal
# In[368]:


test_wp = pd.merge(test_wp, final[['Place', 'ConfirmedCases', 'Fatalities']], on='Place',
                   how='left', validate='m:1')

test_wp
# In[369]:


LAST_DATE

test_wp
# In[370]:


test_wp.ConfirmedCases = np.exp(
    np.log(test_wp.ConfirmedCases + 1) \
    + test_wp.case_slope *
    (test_wp.Date - LAST_DATE).dt.days) - 1

test_wp.Fatalities = np.exp(
    np.log(test_wp.Fatalities + 1) \
    + test_wp.fatality_slope *
    (test_wp.Date - LAST_DATE).dt.days) - 1

# test_wp.Fatalities = np.exp(
#                             np.log(test_wp.ConfirmedCases + 1) \
#                               + test_wp.cfr_pred  )  -1


# In[371]:


LAST_DATE

final[final.Place == 'Italy']
# In[372]:


test_wp[(test_wp.Country == 'Italy')].groupby('Date').sum()[:10]

# In[373]:


test_wp[(test_wp.Country == 'US')].groupby('Date').sum().iloc[-5:]

# In[ ]:


# ### Final Merge

# In[374]:


final = train_bk[train_bk.Date == test.Date.min() - datetime.timedelta(1)]

# In[375]:


final.head()

# In[376]:


test['elapsed'] = (test.Date - final.Date.max()).dt.days

test.Date
# In[377]:


test.elapsed

# In[ ]:


# ### CFR Caps

# In[378]:


full_bk = test_wp.copy()

# In[379]:


full = test_wp.copy()

# In[ ]:


# In[380]:


BASE_RATE = 0.01

# In[381]:


CFR_CAP = 0.13

# In[ ]:


# In[382]:


lplot(full_bk)

# In[383]:


lplot(full_bk, columns=['case_slope', 'fatality_slope'])

# In[ ]:


# In[384]:


full['cfr_imputed_fatalities_low'] = full.ConfirmedCases * np.exp(full.cfr_pred) / np.exp(0.5)
full['cfr_imputed_fatalities_high'] = full.ConfirmedCases * np.exp(full.cfr_pred) * np.exp(0.5)
full['cfr_imputed_fatalities'] = full.ConfirmedCases * np.exp(full.cfr_pred)

# In[ ]:


full[(full.case_slope > 0.02) &
     (full.Fatalities < full.cfr_imputed_fatalities_low) &
     (full.cfr_imputed_fatalities_low > 0.3) &
     (full.Fatalities < 100) &
     (full.Country != 'China')] \
    .groupby('Place').count() \
    .sort_values('ConfirmedCases', ascending=False).iloc[:, 9:]
# In[385]:


full[(full.case_slope > 0.02) &
     (full.Fatalities < full.cfr_imputed_fatalities_low) &
     (full.cfr_imputed_fatalities_low > 0.3) &
     (full.Fatalities < 100000) &
     (full.Country != 'China') &
     (full.Date == datetime.datetime(2020, 4, 15))] \
    .groupby('Place').last() \
    .sort_values('Fatalities', ascending=False).iloc[:, 9:]

# In[386]:


(np.log(full.Fatalities + 1) - np.log(full.cfr_imputed_fatalities)).plot(kind='hist', bins=250)

full[
    (np.log(full.Fatalities + 1) < np.log(full.cfr_imputed_fatalities_high + 1) - 0.5)
    & (~full.Country.isin(['China', 'Korea, South']))
    ][full.Date == train.Date.max()] \
    .groupby('Place').first() \
    .sort_values('cfr_imputed_fatalities', ascending=False).iloc[:, 9:]
# In[387]:


full[(full.case_slope > 0.02) &
     (full.Fatalities < full.cfr_imputed_fatalities_low) &
     (full.cfr_imputed_fatalities_low > 0.3) &
     (full.Fatalities < 100000) &
     (~full.Country.isin(['China', 'Korea, South']))][full.Date == train.Date.max()] \
    .groupby('Place').first() \
    .sort_values('cfr_imputed_fatalities', ascending=False).iloc[:, 9:]

# In[388]:


full.Fatalities = np.where(
    (full.case_slope > 0.02) &
    (full.Fatalities <= full.cfr_imputed_fatalities_low) &
    (full.cfr_imputed_fatalities_low > 0.3) &
    (full.Fatalities < 100000) &
    (~full.Country.isin(['China', 'Korea, South'])),

    (full.cfr_imputed_fatalities_high + full.cfr_imputed_fatalities) / 2,
    full.Fatalities)

assert len(full) == len(data_wp)
x_test.shape
# In[389]:


full['elapsed'] = (test_wp.Date - LAST_DATE).dt.days

# In[390]:


full[(full.case_slope > 0.02) &
     (np.log(full.Fatalities + 1) < np.log(full.ConfirmedCases * BASE_RATE + 1) - 0.5) &
     (full.Country != 'China')] \
    [full.Date == datetime.datetime(2020, 4, 5)] \
    .groupby('Place').last().sort_values('ConfirmedCases', ascending=False).iloc[:, 8:]

full.Fatalities.max()
# In[391]:


full.Fatalities = np.where((full.case_slope > 0.02) &
                           (full.Fatalities < full.ConfirmedCases * BASE_RATE) &
                           (full.Country != 'China'),

                           np.exp(
                               np.log(full.ConfirmedCases * BASE_RATE + 1) \
                               * np.clip(0.5 * (full.elapsed - 1) / 30, 0, 1) \
 \
                               + np.log(full.Fatalities + 1) \
                               * np.clip(1 - 0.5 * (full.elapsed - 1) / 30, 0, 1)
                           ) - 1

                           ,
                           full.Fatalities)

full.elapsed
# In[392]:


full[(full.case_slope > 0.02) &
     (full.Fatalities > full.cfr_imputed_fatalities_high) &
     (full.cfr_imputed_fatalities_low > 0.4) &
     (full.Country != 'China')] \
    .groupby('Place').count() \
    .sort_values('ConfirmedCases', ascending=False).iloc[:, 8:]

full[full.Place == 'United KingdomTurks and Caicos Islands']
# In[393]:


full[(full.case_slope > 0.02) &
     (full.Fatalities > full.cfr_imputed_fatalities_high * 2) &
     (full.cfr_imputed_fatalities_low > 0.4) &
     (full.Country != 'China')] \
    .groupby('Place').last() \
    .sort_values('ConfirmedCases', ascending=False).iloc[:, 8:]

# In[394]:


full[(full.case_slope > 0.02) &
     (full.Fatalities > full.cfr_imputed_fatalities_high * 1.5) &
     (full.cfr_imputed_fatalities_low > 0.4) &
     (full.Country != 'China')][full.Date == train.Date.max()] \
    .groupby('Place').first() \
    .sort_values('ConfirmedCases', ascending=False).iloc[:, 8:]

# In[ ]:


# In[395]:


full.Fatalities = np.where((full.case_slope > 0.02) &
                           (full.Fatalities > full.cfr_imputed_fatalities_high * 2) &
                           (full.cfr_imputed_fatalities_low > 0.4) &
                           (full.Country != 'China'),

                           full.cfr_imputed_fatalities,

                           full.Fatalities)

full.Fatalities = np.where((full.case_slope > 0.02) &
                           (full.Fatalities > full.cfr_imputed_fatalities_high) &
                           (full.cfr_imputed_fatalities_low > 0.4) &
                           (full.Country != 'China'),
                           np.exp(
                               0.6667 * np.log(full.Fatalities + 1) \
                               + 0.3333 * np.log(full.cfr_imputed_fatalities + 1)
                           ) - 1,

                           full.Fatalities)

# In[ ]:


# In[396]:


full[(full.Fatalities > full.ConfirmedCases * CFR_CAP) &
     (full.ConfirmedCases > 1000)

     ].groupby('Place').last().sort_values('Fatalities', ascending=False)

full.Fatalities = np.where((full.Fatalities > full.ConfirmedCases * CFR_CAP) &
                           (full.ConfirmedCases > 1000)
                           ,
                           full.ConfirmedCases * CFR_CAP \
                           * np.clip((full.elapsed - 5) / 15, 0, 1) \
                           + full.Fatalities * np.clip(1 - (full.elapsed - 5) / 15, 0, 1)
                           ,
                           full.Fatalities)
train[train.Country == 'Italy']
final[final.Country == 'US'].sum()
# In[397]:


(np.log(full.Fatalities + 1) - np.log(full.cfr_imputed_fatalities)).plot(kind='hist', bins=250)

# In[ ]:


# ### Fix Slopes now
final
# In[398]:


assert len(pd.merge(full, final, on='Place', suffixes=('', '_i'), validate='m:1')) == len(full)

# In[399]:


ffm = pd.merge(full, final, on='Place', suffixes=('', '_i'), validate='m:1')
ffm['fatality_slope'] = (np.log(ffm.Fatalities + 1) - np.log(ffm.Fatalities_i + 1)) / ffm.elapsed
ffm['case_slope'] = (np.log(ffm.ConfirmedCases + 1) - np.log(ffm.ConfirmedCases_i + 1)) / ffm.elapsed

# #### Fix Upward Slopers
final_slope = (ffm.groupby('Place').last().case_slope)
final_slope.sort_values(ascending=False)

high_final_slope = final_slope[final_slope > 0.1].indexslope_change = (
            ffm.groupby('Place').last().case_slope - ffm.groupby('Place').first().case_slope)
slope_change.sort_values(ascending=False)
high_slope_increase = slope_change[slope_change > 0.05].index
# In[ ]:


test.Date.min()
set(high_slope_increase) & set(high_final_slope)
ffm.groupby('Date').case_slope.median()
# In[ ]:


# ### Fix Drop-Offs

# In[400]:


ffm[np.log(ffm.Fatalities + 1) < np.log(ffm.Fatalities_i + 1) - 0.2][
    ['Place', 'Date', 'elapsed', 'Fatalities', 'Fatalities_i']]

# In[401]:


ffm[np.log(ffm.ConfirmedCases + 1) < np.log(ffm.ConfirmedCases_i + 1) - 0.2][
    ['Place', 'elapsed', 'ConfirmedCases', 'ConfirmedCases_i']]

# In[ ]:


(ffm.groupby('Place').last().fatality_slope - ffm.groupby('Place').first().fatality_slope) \
    .sort_values(ascending=False)[:10]
# ### Display

# In[402]:


new_places = train[(train.Date == test.Date.min() - datetime.timedelta(1)) &
                   (train.ConfirmedCases == 0)
                   ].Place

full[full.Country == 'US'].groupby('Date').agg(
    {'ForecastId': 'count',
     'case_slope': 'mean',
     'fatality_slope': 'mean',
     'ConfirmedCases': 'sum',
     'Fatalities': 'sum',
     })
# In[403]:


full_bk[(full_bk.Date == test.Date.max()) &
        (~full_bk.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
     'fatality_slope': 'last',
     'ConfirmedCases': 'sum',
     'Fatalities': 'sum',
     }
).sort_values('ConfirmedCases', ascending=False)

full[full.Country == 'China'].groupby('Date').agg(
    {'ForecastId': 'count',
     'case_slope': 'mean',
     'fatality_slope': 'mean',
     'ConfirmedCases': 'sum',
     'Fatalities': 'sum',
     })[::5]
# In[ ]:


ffc = pd.merge(final, full, on='Place', validate='1:m')
ffc[(np.log(ffc.Fatalities_x) - np.log(ffc.ConfirmedCase_x)) / ffc.elapsed_y]
ffm.groupby('Place').case_slope.last().sort_values(ascending=False)[:30]
lplot(test_wp)
lplot(test_wp, columns=['case_slope', 'fatality_slope'])
# In[ ]:


# In[404]:


lplot(ffm[~ffm.Place.isin(new_places)])

# In[ ]:


# In[405]:


lplot(ffm[~ffm.Place.isin(new_places)], columns=['case_slope', 'fatality_slope'])

# In[ ]:


test.Date.min()
# In[406]:


ffm.fatality_slope = np.clip(ffm.fatality_slope, None, 0.5)

ffm.case_slope = np.clip(ffm.case_slope, None, 0.25)
# In[ ]:


for lr in [0.05, 0.02, 0.01, 0.007, 0.005, 0.003]:
    ffm.loc[(ffm.Place == ffm.Place.shift(1))
            & (ffm.Place == ffm.Place.shift(-1)) &
            (np.abs((ffm.case_slope.shift(-1) + ffm.case_slope.shift(1)) / 2
                    - ffm.case_slope).fillna(0)
             > lr), 'case_slope'] = \
        (ffm.case_slope.shift(-1) + ffm.case_slope.shift(1)) / 2

# In[407]:


for lr in [0.2, 0.14, 0.1, 0.07, 0.05, 0.03, 0.01]:
    ffm.loc[(ffm.Place == ffm.Place.shift(4))
            & (ffm.Place == ffm.Place.shift(-4)), 'fatality_slope'] = \
        (ffm.fatality_slope.shift(-2) * 0.25 \
         + ffm.fatality_slope.shift(-1) * 0.5 \
         + ffm.fatality_slope \
         + ffm.fatality_slope.shift(1) * 0.5 \
         + ffm.fatality_slope.shift(2) * 0.25) / 2.5

# In[ ]:


# In[408]:


ffm.ConfirmedCases = np.exp(
    np.log(ffm.ConfirmedCases_i + 1) \
    + ffm.case_slope *
    ffm.elapsed) - 1

ffm.Fatalities = np.exp(
    np.log(ffm.Fatalities_i + 1) \
    + ffm.fatality_slope *
    ffm.elapsed) - 1
# test_wp.Fatalities = np.exp(
#                             np.log(test_wp.ConfirmedCases + 1) \
#                               + test_wp.cfr_pred  )  -1


# In[ ]:


# In[409]:


lplot(ffm[~ffm.Place.isin(new_places)])

# In[ ]:


# In[410]:


lplot(ffm[~ffm.Place.isin(new_places)], columns=['case_slope', 'fatality_slope'])

# In[ ]:


# In[411]:


ffm[(ffm.Date == test.Date.max()) &
    (~ffm.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
     'fatality_slope': 'last',
     'ConfirmedCases': 'sum',
     'Fatalities': 'sum',
     }
).sort_values('ConfirmedCases', ascending=False)

# In[ ]:


# In[412]:


ffm_bk = ffm.copy()

# In[ ]:


# In[ ]:


# In[413]:


ffm = ffm_bk.copy()

# In[414]:


counter = Counter(data.Place)
# counter.most_common()
median_count = np.median([counter[group] for group in ffm.Place])
# [ (group, np.round( np.power(counter[group] / median_count, -1),3) ) for group in
#      counter.keys()]
c_count = [np.clip(
    np.power(counter[group] / median_count, -1.5), None, 2.5) for group in ffm.Place]

# In[415]:


[(group, np.round(np.power(counter[group] / median_count, -1.5), 3)) for group in
 counter.keys()]

# In[416]:


RATE_MULT = 0.00
RATE_ADD = 0.003
LAG_FALLOFF = 15

ma_factor = np.clip((ffm.elapsed - 14) / 14, 0, 1)

ffm.case_slope = np.where(ffm.elapsed > 0,
                          0.7 * ffm.case_slope * (1 + ma_factor * RATE_MULT) \
                          + 0.3 * (ffm.case_slope.ewm(span=LAG_FALLOFF).mean() \
                                   * np.clip(ma_factor, 0, 1)
                                   + ffm.case_slope * np.clip(1 - ma_factor, 0, 1))

                          + RATE_ADD * ma_factor * c_count,
                          ffm.case_slope)

# --

RATE_MULT = 0
RATE_ADD = 0.015
LAG_FALLOFF = 15

ma_factor = np.clip((ffm.elapsed - 10) / 14, 0, 1)

ffm.fatality_slope = np.where(ffm.elapsed > 0,
                              0.3 * ffm.fatality_slope * (1 + ma_factor * RATE_MULT) \
                              + 0.7 * (ffm.fatality_slope.ewm(span=LAG_FALLOFF).mean() \
                                       * np.clip(ma_factor, 0, 1)
                                       + ffm.fatality_slope * np.clip(1 - ma_factor, 0, 1))

                              + RATE_ADD * ma_factor * c_count \
 \
                              * (ffm.Country != 'China')
                              ,
                              ffm.case_slope)

# In[417]:


ffm.ConfirmedCases = np.exp(
    np.log(ffm.ConfirmedCases_i + 1) \
    + ffm.case_slope *
    ffm.elapsed) - 1

ffm.Fatalities = np.exp(
    np.log(ffm.Fatalities_i + 1) \
    + ffm.fatality_slope *
    ffm.elapsed) - 1
# test_wp.Fatalities = np.exp(
#                             np.log(test_wp.ConfirmedCases + 1) \
#                               + test_wp.cfr_pred  )  -1


# In[ ]:


# In[418]:


lplot(ffm[~ffm.Place.isin(new_places)])

# In[ ]:


# In[419]:


lplot(ffm[~ffm.Place.isin(new_places)], columns=['case_slope', 'fatality_slope'])

# In[ ]:


LAST_DATE
# In[420]:


ffm_bk[(ffm_bk.Date == test.Date.max()) &
       (~ffm_bk.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
     'fatality_slope': 'last',
     'ConfirmedCases': 'sum',
     'Fatalities': 'sum',
     }
).sort_values('ConfirmedCases', ascending=False)[:15]

# In[421]:


ffm[(ffm.Date == test.Date.max()) &
    (~ffm.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
     'fatality_slope': 'last',
     'ConfirmedCases': 'sum',
     'Fatalities': 'sum',
     }
).sort_values('ConfirmedCases', ascending=False)[:15]

# In[ ]:


# In[ ]:


# In[ ]:


# In[422]:


ffm_bk[(ffm_bk.Date == test.Date.max()) &
       (~ffm_bk.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
     'fatality_slope': 'last',
     'ConfirmedCases': 'sum',
     'Fatalities': 'sum',
     }
).sort_values('ConfirmedCases', ascending=False)[-50:]

# In[423]:


ffm[(ffm.Date == test.Date.max()) &
    (~ffm.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
     'fatality_slope': 'last',
     'ConfirmedCases': 'sum',
     'Fatalities': 'sum',
     }
).loc[ffm_bk[(ffm_bk.Date == test.Date.max()) &
             (~ffm_bk.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
     'fatality_slope': 'last',
     'ConfirmedCases': 'sum',
     'Fatalities': 'sum',
     }
).sort_values('ConfirmedCases', ascending=False)[-50:].index]

# In[ ]:


# In[424]:


# use country-specific CFR !!!!  helps cap US and raise up Italy !
# could also use lagged CFR off cases as of 2 weeks ago...
# ****  keep everything within ~0.5 order of magnitude of its predicted CFR.. !!


# In[ ]:


# ### Join
assert len(test_wp) == len(full)
full = pd.merge(test_wp, full[['Place', 'Date', 'Fatalities']], on=['Place', 'Date'],
                validate='1:1')
# In[ ]:


# ### Fill in New Places with Ramp Average

# In[425]:


NUM_TEST_DATES = len(test.Date.unique())

base = np.zeros((2, NUM_TEST_DATES))
base2 = np.zeros((2, NUM_TEST_DATES))

# In[426]:


for idx, c in enumerate(['ConfirmedCases', 'Fatalities']):
    for n in range(0, NUM_TEST_DATES):
        base[idx, n] = np.mean(
            np.log(train[((train.Date < test.Date.min())) &
                         (train.ConfirmedCases > 0)].groupby('Country').nth(n)[c] + 1))

# In[427]:


base = np.pad(base, ((0, 0), (6, 0)), mode='constant', constant_values=0)

# In[428]:


for n in range(0, base2.shape[1]):
    base2[:, n] = np.mean(base[:, n + 0: n + 7], axis=1)

# In[429]:


new_places = train[(train.Date == test.Date.min() - datetime.timedelta(1)) &
                   (train.ConfirmedCases == 0)
                   ].Place

# In[430]:


# fill in new places
ffm.ConfirmedCases = np.where(ffm.Place.isin(new_places),
                              base2[0, (ffm.Date - test.Date.min()).dt.days],
                              ffm.ConfirmedCases)
ffm.Fatalities = np.where(ffm.Place.isin(new_places),
                          base2[1, (ffm.Date - test.Date.min()).dt.days],
                          ffm.Fatalities)

# In[ ]:


# In[431]:


ffm[ffm.Country == 'US'].groupby('Date').agg(
    {'ForecastId': 'count',
     'case_slope': 'mean',
     'fatality_slope': 'mean',
     'ConfirmedCases': 'sum',
     'Fatalities': 'sum',
     })

train[train.Country == 'US'].Province_State.unique()
# ### Save

# In[ ]:


# In[ ]:


# In[432]:


sub = pd.read_csv(path + '/input' + '/submission.csv')

# In[433]:


filename = 'submission_c19_wk4_FINAL_MODEL'

# In[434]:


scl = sub.columns.to_list()

# In[ ]:


# In[435]:


outs = [full_bk[scl] * 0.7 + ffm[scl] * 0.3,
        full_bk[scl] * 0.3 + ffm[scl] * 0.7]
out_names = ['low', 'high']

# In[436]:


for idx, ooox in enumerate(outs):
    file = '{}-{}-{}-{}_{}.csv'.format(filename, MODEL_Y, q, out_names[idx], pp)
    out = outs[idx]

    out = out[sub.columns.to_list()]
    out.ForecastId = np.round(out.ForecastId, 0).astype(int)
    out = np.round(out, 2)

    out.to_csv(path + '/output' + '/' + file, index=False)

# In[437]:


file


# ### Score

# In[ ]:


# In[438]:


def RMSLE(df, verbose=0, max_date=None):
    def score(scores):
        return np.sqrt(np.sum(np.power(scores, 2)) / len(scores))

    if max_date is not None:
        df = df[df.Date <= max_date]

    all_scores = []
    for col in ['Fatalities', 'ConfirmedCases']:
        scores = np.log(df[col + '_true'] + 1) - np.log(df[col + '_pred'] + 1)
        df[col[:1] + '_error'] = scores

        print("{}: {:.2f}".format(col, score(scores)))
        all_scores.append(score(scores))

        print("   pos: {:.2f}".format(score(np.clip(scores, 0, None))))
        print("   neg: {:.2f}".format(score(np.clip(scores, None, 0))))

        if verbose > 1:
            print(np.round(scores.reset_index().drop(columns='Place').groupby('Date').
                           apply(score), 2))
            #        print(scores.reset_index().groupby('Date').apply(score))
            #       print(scores.groupby(df.reset_index().Date).apply(score))
            print()

    #     print()
    print("      avg: {:.3f}\n".format(np.mean(all_scores)))
    return df


train_bk
# In[439]:


for arr in [ffm_bk, ffm]:
    combo = pd.merge(train_bk, ffm, on=['Place', 'Date'], suffixes=['_true', '_pred'])
    combo = combo[combo.Place.isin(list(final.Place[(final.ConfirmedCases > 0)]))]
    combo.set_index(['Place', 'Date'], inplace=True)
    combo_scored = RMSLE(combo, verbose=0)
    combo_scored = np.round(combo_scored, 1)

# new data: 0.45/0.33 for 0.390; 0.385;
# with 0.2 frac and with skew to smaller ones 0.379 now; (0.42/0.34)
# 0.42/0.33;  0.377 -- other peopl's models were 0.409 and clearly worse on each side.


# In[ ]:


# In[ ]:


# In[ ]:


# In[440]:


# old:

## new data load 4-4: priors, actions, single model:  0.43 / 0.29;; many: 0.39 / 0.30, ibid.
## my own: 0.39 / 0.31 once sufficiently bagged      priors gets worse with vopani in
# my own: 0.38 / 0.31 with bags
##  0.38 / 0.29 (!!!) with nearest features added !!!  0.337 . 0.355, 0.344
## fresh 0.38 / 0.31: 0.347 (dropping country and containment entirely)
## CFR vs world and nearby! (and drop raw _cfr), down to 0.339 (may or may not hold...)


# In[ ]:


prior_best = pd.read_csv(path + '/output/final_blend_1' + '/' + 'submission_basic_lgb_joined.csv')
prior_best_full = pd.merge(prior_best, test, on='ForecastId')
combo_prior_best = pd.merge(full_train, prior_best_full,
                            on=['Place', 'Date'], suffixes=['_true', '_pred'])
combo_prior_best.set_index(['Place', 'Date'], inplace=True)
combo_prior_best_scored = RMSLE(combo_prior_best, verbose=0)
combo_prior_best_scored = np.round(combo_prior_best_scored, 1)


Benchmark: test = test[test.Date >= test_dates[3]]

ffm[ffm.Date == test.Date.max()].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'mean',
     'fatality_slope': 'mean',
     'ConfirmedCases': 'sum',
     'Fatalities': 'sum',
     }
).sort_values('ConfirmedCases', ascending=False)
ffm[ffm.Date == test.Date.max()].groupby('Place').agg(
    {'ForecastId': 'count',
     'case_slope': 'mean',
     'fatality_slope': 'mean',
     'ConfirmedCases': 'sum',
     'Fatalities': 'sum',
     }
).sort_values('ConfirmedCases', ascending=False)
# In[ ]:


# In[442]:


# floor deaths at ~3-4 (or population-based)
# push developing world up 2x-3x at tails
# probably lighten major countries by 0.67x;

# push developing much higher as other sub (8x, floors at ~30/6), major countries as-is


# In[ ]:


np.exp([0, 1, 2, 3, 4]) - 1
# In[443]:


# guess higher on developing world by 2-3x;
# cut US by a bit (or lockdowns will) from ~120k


# In[ ]:


# In[444]:


combo_scored = combo_scored.reset_index()

# In[ ]:


combo_scored[combo_scored.Place.isin(
    list(final.Place[(final.ConfirmedCases > 0)])
)]
# In[ ]:


# In[445]:


finalp = final.set_index('Place')

# In[446]:


combo_scored.set_index(['Place', 'Date'], inplace=True)

final.Place[(final.ConfirmedCases > 0)]
# In[ ]:


# In[ ]:


# In[447]:


key_cols = [c for c in combo_scored.columns.to_list() if
            c is 'Date' or any(z in c for z in
                               ['Place', 'Fatal', 'Cases', 'error']

                               ) and 'Rate' not in c
            ]

# In[448]:


list.sort(key_cols)

# In[ ]:


# In[449]:


short_names = {'ConfirmedCases_pred': 'ConfCases_pred',
               'ConfirmedCases_true': 'ConfCases_true'}

# In[450]:


# before: SAfr, Kuw, Rom, Qatar ~1


# In[451]:


# *** PLACES VERY SLOW TO REPORT FIRST FATALITY -- LIGHTGBM MAY DOMINATE HERE


# In[ ]:


# In[452]:


combo_scored.sort_values('F_error', ascending=True)[key_cols].rename(columns=short_names)[:15]

# In[453]:


pd.merge(finalp,
         combo_scored[key_cols].groupby('Place').F_error.apply(
             lambda x: np.power(np.clip(x, None, 0), 2).sum()) \
         , on='Place').sort_values('F_error', ascending=False)[:10]

# In[ ]:


# In[454]:


combo_scored.sort_values('F_error', ascending=False)[key_cols].rename(columns=short_names)[:20]

# In[455]:


pd.merge(finalp,
         combo_scored[key_cols].groupby('Place').F_error.apply(
             lambda x: np.power(np.clip(x, 0, None), 2).sum()) \
         , on='Place').sort_values('F_error', ascending=False)[:20]

# In[ ]:


# In[456]:


# GDP per capita, total GDP, etc.
# then some measure of Asia (better at slowing things, at all stages)
# any metric of restrictions having gone into place already


# In[ ]:


# In[ ]:


# #### Case Error

# In[457]:


combo_scored.sort_values('C_error', ascending=True)[key_cols].rename(columns=short_names)[:20]

# In[458]:


pd.merge(finalp,
         combo_scored[key_cols].groupby('Place').C_error.apply(
             lambda x: np.power(np.clip(x, None, 0), 2).sum()) \
         , on='Place').sort_values('C_error', ascending=False)[:10]

pd.merge(finalp,
         combo_scored[key_cols].groupby('Place').C_error.apply(
             lambda x: np.power(np.clip(x, None, 0), 2).sum()) \
         , on='Place').sum()
# In[ ]:


# In[459]:


combo_scored.sort_values('C_error', ascending=False)[key_cols].rename(columns=short_names)[:20]

# In[460]:


pd.merge(finalp,
         combo_scored[key_cols].groupby('Place').C_error.apply(
             lambda x: np.power(np.clip(x, 0, None), 2).sum()) \
         , on='Place').sort_values('C_error', ascending=False)[:15]

# In[ ]:


combo_scored
# In[461]:


combo

# In[462]:


# pd.merge(finalp,
combo_scored.groupby('Country_true').C_error.apply(
    lambda x: np.power(np.clip(x, 0, None), 2).sum()) \
    .sort_values(ascending=False)[:20] \
    #         , on='Country').sort_values('C_error', ascending=False)[:15]

# In[ ]:


# ### All Errors

# In[463]:


# build features to spot that US leap--any clues for these?
# correct any China unflatteners
# BUILD FOR DEPLOYMENT OF ACTUAL PREDICTIONS EITHER WAY
# look at stepped two days forward--how does it do then
# if it's all those US ramps--why do the errors build so slowly??


# ##### Total

# In[464]:


# pd.merge(finalp,
combo_scored.groupby('Country_true').C_error.apply(
    lambda x: np.power(np.clip(x, None, None), 2).sum()) \
    .sort_values(ascending=False)[:20] \
    #         , on='Country').sort_values('C_error', ascending=False)[:15]

# In[465]:


# pd.merge(finalp,
combo_scored.groupby('Country_true').F_error.apply(
    lambda x: np.power(np.clip(x, None, None), 2).sum()) \
    .sort_values(ascending=False)[:20] \
    #         , on='Country').sort_values('C_error', ascending=False)[:15]

# In[ ]:


# ##### Avg Error

# In[466]:


counter = Counter(places)
[(p, counter[p]) for p in combo_scored.groupby('Country_true').C_error.apply(
    lambda x: np.power(np.clip(x, None, None), 2).mean()) \
                              .sort_values(ascending=False)[:20].index.to_list()]

# In[467]:


# pd.merge(finalp,
combo_scored.groupby('Country_true').C_error.apply(
    lambda x: np.power(np.clip(x, None, None), 2).mean()) \
    .sort_values(ascending=False)[:20] \
    #         , on='Country').sort_values('C_error', ascending=False)[:15]

Country_true
Mali
1.64625
Kazakhstan
0.98875
Grenada
0.76125
Guinea
0.5725
Niger
0.49125
Libya
0.48625
Suriname
0.4375
United
Kingdom
0.26053571
Guyana
0.24875
Guinea - Bissau
0.24375
# In[ ]:


# In[468]:


np.median(list(counter.values()))

# In[469]:


counter = Counter(places)
[(p, counter[p]) for p in combo_scored.groupby('Country_true').F_error.apply(
    lambda x: np.power(np.clip(x, None, None), 2).mean()) \
                              .sort_values(ascending=False)[:20].index.to_list()]



# NUMBER ONE NEED IS TO SPOT WHO MIGHT RAMP VERY QUICKLY
# i.e. whose absolute count will catch up to population ratio to other major areas...

# part of this is "from 0" and "US acceleration" but a lot of it isn't


# simplest is maintain ~7day rate
# next simplest is maintain ~7d rate, with a daily decay factor (i.e. log of simple lin reg)
# next is predict decay rate to apply, based on current and properties, etc.


file = file.replace('40', '50')


for o_name in out_names:
    filename = file.replace(out_names[-1], o_name)
    # file = 'submission_basic_lgb_private.csv'
    try:
        private = pd.read_csv(path + '/output' + '/' + filename.replace('public', 'private'))
        public = pd.read_csv(path + '/output' + '/' + filename.replace('private', 'public'))

        full_pred = pd.concat((private, public[~public.ForecastId.isin(private.ForecastId)]),
                              ignore_index=True).sort_values('ForecastId')

        full_pred.to_csv(path + '/output' + '/' + filename.replace('private', 'joined').replace('public', 'joined'),
                         index=False)
    except:
        pass;


# LGB:   239k US;  124k Spain (high); 76k UK; 67k Italy, maybe chop everything by ~1.5 and ~5
# Blend:

pd.merge(private, test[test.Date == test.Date.max()], on=['ForecastId']) \
    .groupby('Country').sum() \
    .sort_values('Fatalities', ascending=False)

# mainly need to stop relying on a single slope... replace with others nearby
# over-relies on fatality etc.; better to have demographics and similar etc.

for file_suffix in ['controlled_joined', 'extended_joined']:
    blend_dir = path + '/output/'

    sets = []
    for file in [file for file in os.listdir(blend_dir) if '{}.csv'.format(file_suffix) in file]:
        sets.append(pd.read_csv(blend_dir + '/' + file))
        print(file)

    if len(sets):
        for s in sets:
            s[s.ForecastId % 43 == 0].sum()

        # combine
        p = sets[0]
        for s in sets[1:]:
            p = pd.merge(p, s, on='ForecastId')

        cols = ['ConfirmedCases', 'Fatalities']
        for col in cols:
            p[col] = np.mean(
                np.log(p[[c for c in p if col in c]] + 1), axis=1)

        blend = p[['ForecastId'] + cols]
        predicted = pd.merge(blend, test, on='ForecastId')

        r1 = predicted[predicted.Date == predicted.Date.max()].groupby('Country')[cols].apply(
            lambda x: (np.exp(x) - 1).sum()).sort_values('ConfirmedCases', ascending=False)
        r1.head()
        r1.tail()

        r2 = predicted[predicted.Date == predicted.Date.max()].groupby('Country')[cols].apply(
            lambda x: (np.exp(x) - 1).sum()).sort_values('Fatalities', ascending=False)
        r2.head()
        r2.tail()

        adj_pred = blend.copy()
        adj_pred.ConfirmedCases = np.exp(adj_pred.ConfirmedCases) - 1
        adj_pred.Fatalities = np.exp(adj_pred.Fatalities) - 1
        np.round(adj_pred, 2).to_csv(blend_dir + '/{}_BLENDED.csv'.format(file_suffix), index=False)


ihme = pd.read_csv(path + '/outside_data' + '/Hospitalization_all_locs.csv')

ihme.date = pd.to_datetime(ihme.date)

i_joined = pd.merge(pred[(pred.Country == 'US') & (pred.Date > train_bk.Date.max())],
                    ihme[['location', 'date', 'totdea_mean']] \
                    .rename(columns={'location': 'Province_State',
                                     'totdea_mean': 'IHME_f',
                                     'date': 'Date'}, ),
                    on=['Date', 'Province_State'],
                    how='inner', validate='1:1')
i_joined['elapsed'] = (i_joined.Date - train_bk.Date.max()).dt.days
predicted = i_joined.copy()
cols = ['ConfirmedCases', 'Fatalities', 'IHME_f']
for col in [c for c in predicted.columns.to_list() if any(z in c for z in cols)]:
    predicted[col] = np.log(predicted[col] + 1)
    predicted['gap'] = (predicted.IHME_f - predicted.Fatalities) \
                       * np.clip(predicted.elapsed / 25, 0, 1)
    predicted.gap = np.where(predicted.gap < 0,
                             predicted.gap * 0.7,
                             predicted.gap)
    predicted['Fatalities_a'] = predicted.Fatalities + predicted.gap * 0.43
predicted['ConfirmedCases_a'] = predicted.ConfirmedCases + predicted.gap * 0.2
cols_2 = cols + ['ConfirmedCases_a', 'Fatalities_a']
np.round(predicted[predicted.Date == predicted.Date.max()].groupby('Place') \
             [cols_2 + ['gap']].apply(lambda x: (np.exp(x) - 1).sum()) \
         .sort_values('ConfirmedCases', ascending=False), 1)[:51]
adj_pred = predicted.copy()

for col in ['ConfirmedCases', 'ConfirmedCases_a', 'Fatalities', 'Fatalities_a', 'IHME_f']:
    adj_pred[col] = np.exp(adj_pred[col]) - 1
    np.round(adj_pred.groupby('Place').last(), 1)
    np.round(adj_pred.groupby('Date').sum(), 1)
    adj_pred[
        adj_pred.Date == adj_pred.Date.max()].sum()  ##### Outputcols = ['ConfirmedCases', 'Fatalities']adj_pred = adj_pred[['ForecastId', 'Fatalities_a', 'ConfirmedCases_a']]\
    .rename(columns={'Fatalities_a': 'Fatalities',
                     'ConfirmedCases_a': 'ConfirmedCases'}) \
        #                     [['ForecastId'] + cols]orig_pred = pred[~pred.ForecastId.isin(adj_pred.ForecastId.unique())]len(orig_pred)final_output = np.round(pd.concat((adj_pred[['ForecastId'] + cols],
    orig_pred[['ForecastId'] + cols])) \
            .sort_values('ForecastId'), 2)stamp = str(datetime.datetime.now())[:16].replace(' ', '-').replace(':', "")
    final_output.to_csv(blend_dir + '/' +
                        'combined_model_average_plus_ihme_{}.csv'.format(stamp),
                        index=False)
    # In[482]:

    # assert 0==1

    # In[483]:

    # a = dfsdf

    # ### Final Day Tweaks

    # In[531]:

    blend_dir = path + '/output/'
    files = os.listdir(blend_dir)
    # files = [f for f in files if 'joined' in f]
    files = [f for f in files if '.csv' in f]
    for file in files:
        print(file)

    # In[532]:

    file = files[0]
    # file = 'submission_c19_wk3_FINAL_MODEL-agg_dff-50-low_joined_UPDATED.csv'
    print(file)

    pred = pd.read_csv(blend_dir + '/' + file)
    pred_orig = pred.copy()

    fname = '../sub/' + mname + '.csv'
    pred.to_csv(fname, index=False)
    print(pred)

    # In[533]:

    # data scraped from https://www.worldometers.info/coronavirus/, including past daily snapshots
    # wm = pd.read_csv('wmc.csv', parse_dates=["Date"] , engine="python")
    wm = pd.read_csv('../wmc.csv')
    wm['Date'] = pd.to_datetime(wm.Date)
    cp = ['Country_Region', 'Province_State']
    wm[cp] = wm[cp].fillna('')
    print(wm)
    wm.Date.value_counts()[:10]

    # In[534]:

    # final day adjustment as per northquay
    pname = mname
    # pred = sub.copy()
    # pred = pd.read_csv('sub/' + mname + '.csv')

    # pname = 'kaz0m'
    # pred = pd.read_csv('../week3/sub/'+pname+'.csv')

    # pred_orig = pred.copy()

    # if prev_test:
    #     test = pd.read_csv('../'+pw+'/test.csv')
    # else:
    #     test = pd.read_csv('test.csv')

    test = pd.read_csv('test.csv')
    test[cp] = test[cp].fillna('')

    test.Date = pd.to_datetime(test.Date)
    # train.Date = pd.to_datetime(train.Date)

    # TODAY = datetime.datetime(  *datetime.datetime.today().timetuple()[:3] )
    # TODAY = datetime.datetime(2020, 4, 8)

    print(TODAY)

    final_day = wm[wm.Date == TODAY].copy()
    final_day['cases_final'] = np.expm1(final_day.TotalCases)
    final_day['cases_chg'] = np.expm1(final_day.NewCases)
    final_day['deaths_final'] = np.expm1(final_day.TotalDeaths)
    final_day['deaths_chg'] = np.expm1(final_day.NewDeaths)

    # test.rename(columns={'Country_Region': 'Country'}, inplace=True)
    # test['Place'] = test.Country +  test.Province_State.fillna("")

    # final_day = pd.read_excel(path + '../week3/nq/' + 'final_day.xlsx')
    # final_day = final_day.iloc[1:, :5]
    # final_day = final_day.fillna(0)
    # final_day.columns = ['Country', 'cases_final', 'cases_chg',
    #                      'deaths_final', 'deaths_chg']

    final_day = final_day[['Country_Region', 'Province_State', 'cases_final', 'cases_chg',
                           'deaths_final', 'deaths_chg']].fillna(0)
    # final_day = final_day.drop('Date', axis=1).reset_index(drop=True)
    final_day = final_day.sort_values('cases_final', ascending=False)

    print()
    print('final_day')
    print(final_day.head(n=10), final_day.shape)

    # final_day.Country.replace({'Taiwan': 'Taiwan*',
    #                            'S. Korea': 'Korea, South',
    #                            'Myanmar': 'Burma',
    #                            'Vatican City': 'Holy See',
    #                            'Ivory Coast':  "Cote d'Ivoire",

    #                           },
    #                          inplace=True)

    pred = pd.merge(pred, test, how='left', on='ForecastId')
    print()
    print('pred')
    print(pred.head(n=10), pred.shape)

    # pred = pd.merge(pred, test[test.Province_State.isnull()], how='left', on='ForecastId')

    # compare = pd.merge(pred[pred.Date == TODAY], final_day, on= [ 'Country'],
    #                            validate='1:1')

    compare = pd.merge(pred[pred.Date == TODAY], final_day, on=cp, validate='1:1')

    compare['c_li'] = np.round(np.log(compare.cases_final + 1) - np.log(compare.ConfirmedCases + 1), 2)
    compare['f_li'] = np.round(np.log(compare.deaths_final + 1) - np.log(compare.Fatalities + 1), 2)

    print()
    print('compare')
    print(compare.head(n=10), compare.shape)
    print(compare.describe())

    # compare[compare.c_li > 0.3][['Country', 'ConfirmedCases', 'Fatalities',
    #                                         'cases_final', 'cases_chg',
    #                                     'deaths_final', 'deaths_chg',
    #                                             'c_li', 'f_li']]

    # compare[compare.c_li > 0.15][['Country', 'ConfirmedCases', 'Fatalities',
    #                                         'cases_final', 'cases_chg',
    #                                     'deaths_final', 'deaths_chg',
    #                                             'c_li', 'f_li']]

    # compare[compare.f_li > 0.3][['Country', 'ConfirmedCases', 'Fatalities',
    #                                         'cases_final', 'cases_chg',
    #                                     'deaths_final', 'deaths_chg',
    #                                             'c_li', 'f_li']]

    # compare[compare.f_li > 0.15][['Country', 'ConfirmedCases', 'Fatalities',
    #                                         'cases_final', 'cases_chg',
    #                                     'deaths_final', 'deaths_chg',
    #                                             'c_li', 'f_li']]

    # compare[compare.c_li < -0.15][['Country', 'ConfirmedCases', 'Fatalities',
    #                                         'cases_final', 'cases_chg',
    #                                     'deaths_final', 'deaths_chg',
    #                                             'c_li', 'f_li']]

    # compare[compare.f_li < -0.2][['Country', 'ConfirmedCases', 'Fatalities',
    #                                         'cases_final', 'cases_chg',
    #                                     'deaths_final', 'deaths_chg',
    #                                             'c_li', 'f_li']]

    fixes = pd.merge(pred[pred.Date >= TODAY],
                     compare[cp + ['c_li', 'f_li']], on=cp)

    fixes['c_li'] = np.where(fixes.c_li < 0,
                             0,
                             fixes.c_li)
    fixes['f_li'] = np.where(fixes.f_li < 0,
                             0,
                             fixes.f_li)

    fixes['total_fixes'] = fixes.c_li ** 2 + fixes.f_li ** 2

    print()
    print('most fixes')
    print(fixes.groupby(cp).last().sort_values(['total_fixes', 'Date'], ascending=False).head(n=10))

    # adjustment
    fixes['Fatalities'] = np.round(np.exp((np.log(fixes.Fatalities + 1) + fixes.f_li)) - 1, 3)
    fixes['ConfirmedCases'] = np.round(np.exp((np.log(fixes.ConfirmedCases + 1) + fixes.c_li)) - 1, 3)

    fix_ids = fixes.ForecastId.unique()
    len(fix_ids)

    cols = ['ForecastId', 'ConfirmedCases', 'Fatalities']

    fixed = pd.concat((pred.loc[~pred.ForecastId.isin(fix_ids), cols],
                       fixes[cols])).sort_values('ForecastId')

    # fixed.head()
    # fixed.tail()

    # len(pred_orig)
    # len(fixed)

    # In[535]:

    # most fixes
    #                                     ForecastId  ConfirmedCases  Fatalities  \
    # Country_Region      Province_State
    # Jamaica                                   6364          198.86        8.19
    # Gabon                                     5375          175.06        3.18
    # US                  Maryland             11137        19718.96      970.81
    # Timor-Leste                              10019            31.4        1.54
    # France              Martinique            5031          347.38       12.16
    # Congo (Brazzaville)                       3827          152.08       11.82
    # Barbados                                   946          135.79        9.33
    # US                  New Hampshire        11524         2112.06       75.05
    # Tanzania                                  9933          191.87       10.53
    # Burma                                     1505          265.35       23.28

    #                                          Date  c_li  f_li  total_fixes
    # Country_Region      Province_State
    # Jamaica                            2020-05-14  0.26  0.11       0.0797
    # Gabon                              2020-05-14  0.25   0.0       0.0625
    # US                  Maryland       2020-05-14  0.18  0.14        0.052
    # Timor-Leste                        2020-05-14  0.21   0.0       0.0441
    # France              Martinique     2020-05-14   0.0  0.18       0.0324
    # Congo (Brazzaville)                2020-05-14  0.12   0.0       0.0144
    # Barbados                           2020-05-14   0.0  0.09       0.0081
    # US                  New Hampshire  2020-05-14  0.08   0.0       0.0064
    # Tanzania                           2020-05-14  0.07   0.0       0.0049
    # Burma                              2020-05-14  0.06   0.0       0.0036

    # In[536]:

    fname = blend_dir + '/' + file[:-4] + '_UPDATED_PS.csv'
    fixed.to_csv(fname, index=False)
    print(fname, fixed.shape)

    # In[537]:

    fname = '../sub/' + mname + '_updated.csv'
    fixed.to_csv(fname, index=False)
    print(fname, fixed.shape)

    # In[538]:

    fixed.describe()

    # In[539]:

    fixed[5:10]

    # In[ ]:




