
# coding: utf-8

# In[508]:


import numpy as np
import pandas as pd
import datetime
import glob
import os.path
from pandas.compat import StringIO


# ### NREL Bird Model implementation: for obtaining clear sky GHI

# In[509]:


import itertools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#get_ipython().magic('matplotlib inline')
sns.set_color_codes()


# In[510]:


import pvlib
from pvlib import clearsky, atmosphere
from pvlib.location import Location


# In[511]:


### CONFIGURE RUNS
run_train = False# Disables training & processing of train set; Set it to True for the first time to create a model
#test_location = "Bondville" #Folder name
#test_location = "Boulder" #Folder name
#test_location = "Desert_Rock" #Folder name
#test_location = "Fort_Peck" #Folder name
test_location = "Goodwin_Creek" #Folder name
#test_location = "Penn_State" #Folder name
#test_location = "Sioux_Falls" #Folder name

# All_locations
#bvl = Location(40.1134,-88.3695, 'US/Central', 217.932, 'Bondville')
#bvl = Location(40.0150,-105.2705, 'US/Mountain', 1655.064, 'Boulder')
#bvl = Location(36.621,-116.043, 'US/Pacific', 1010.1072, 'Desert Rock')
#bvl = Location(48,-106.449, 'US/Mountain', 630.0216, 'Fort Peck')
bvl = Location(34.2487,-89.8925, 'US/Central', 98, 'Goodwin Creek')
# bvl = Location(40.798,-77.859, 'US/Eastern', 351.74, 'Penn State')
# bvl = Location(43.544,-96.73, 'US/Central', 448.086, 'Sioux Falls')



test_year = "2017"


# TEST year 2009
#times = pd.DatetimeIndex(start='2009-01-01', end='2010-01-01', freq='1min',tz=bvl.tz)   # 12 months
#  TEST year 2015
#times = pd.DatetimeIndex(start='2015-01-01', end='2016-01-01', freq='1min',tz=bvl.tz)   # 12 months 
# TEST year 2016
#times = pd.DatetimeIndex(start='2016-01-01', end='2017-01-01', freq='1min',tz=bvl.tz)   # 12 months 
# Test year 2017
times = pd.DatetimeIndex(start='2017-01-01', end='2018-01-01', freq='1min',tz=bvl.tz)   # 12 months 


# In[512]:


if run_train:
   # TRAIN set
   times2010and2011 = pd.DatetimeIndex(start='2010-01-01', end='2012-01-01', freq='1min',
                           tz=bvl.tz)   # 24 months of 2010 and 2011 - For training
   cs_2010and2011 = bvl.get_clearsky(times2010and2011) # ineichen with climatology table by default
   cs_2010and2011.drop(['dni','dhi'],axis=1, inplace=True) #updating the same dataframe by dropping two columns
   cs_2010and2011.reset_index(inplace=True)

   cs_2010and2011['index']=cs_2010and2011['index'].apply(lambda x:x.to_datetime())
   cs_2010and2011['year'] = cs_2010and2011['index'].apply(lambda x:x.year)
   cs_2010and2011['month'] = cs_2010and2011['index'].apply(lambda x:x.month)
   cs_2010and2011['day'] = cs_2010and2011['index'].apply(lambda x:x.day)
   cs_2010and2011['hour'] = cs_2010and2011['index'].apply(lambda x:x.hour)
   cs_2010and2011['min'] = cs_2010and2011['index'].apply(lambda x:x.minute)


   cs_2010and2011.drop(cs_2010and2011.index[-1], inplace=True)
   print(cs_2010and2011.shape)
   cs_2010and2011.head()


# In[513]:


# TEST set


cs_test = bvl.get_clearsky(times)
cs_test.drop(['dni','dhi'],axis=1, inplace=True) #updating the same dataframe by dropping two columns
cs_test.reset_index(inplace=True)

cs_test['index']=cs_test['index'].apply(lambda x:x.to_datetime())
cs_test['year'] = cs_test['index'].apply(lambda x:x.year)
cs_test['month'] = cs_test['index'].apply(lambda x:x.month)
cs_test['day'] = cs_test['index'].apply(lambda x:x.day)
cs_test['hour'] = cs_test['index'].apply(lambda x:x.hour)
cs_test['min'] = cs_test['index'].apply(lambda x:x.minute)

cs_test.drop(cs_test.index[-1], inplace=True)
print(cs_test.shape)


# ### Import files from each year in a separate dataframe

# 
# - year            integer	 year, i.e., 1995
# - jday            integer	 Julian day (1 through 365 [or 366])
# - month           integer	 number of the month (1-12)
# - day             integer	 day of the month(1-31)
# - hour            integer	 hour of the day (0-23)
# - min             integer	 minute of the hour (0-59)
# - dt              real	 decimal time (hour.decimalminutes, e.g., 23.5 = 2330)
# - zen             real	 solar zenith angle (degrees)
# - dw_solar        real	 downwelling global solar (Watts m^-2)
# - uw_solar        real	 upwelling global solar (Watts m^-2)
# - direct_n        real	 direct-normal solar (Watts m^-2)
# - diffuse         real	 downwelling diffuse solar (Watts m^-2)
# - dw_ir           real	 downwelling thermal infrared (Watts m^-2)
# - dw_casetemp     real	 downwelling IR case temp. (K)
# - dw_dometemp     real	 downwelling IR dome temp. (K)
# - uw_ir           real	 upwelling thermal infrared (Watts m^-2)
# - uw_casetemp     real	 upwelling IR case temp. (K)
# - uw_dometemp     real	 upwelling IR dome temp. (K)
# - uvb             real	 global UVB (milliWatts m^-2)
# - par             real	 photosynthetically active radiation (Watts m^-2)
# - netsolar        real	 net solar (dw_solar - uw_solar) (Watts m^-2)
# - netir           real	 net infrared (dw_ir - uw_ir) (Watts m^-2)
# - totalnet        real	 net radiation (netsolar+netir) (Watts m^-2)
# - temp            real	 10-meter air temperature (?C)
# - rh              real	 relative humidity (%)
# - windspd         real	 wind speed (ms^-1)
# - winddir         real	 wind direction (degrees, clockwise from north)
# - pressure        real	 station pressure (mb)
# 

# In[514]:


cols = ['year', 'jday', 'month', 'day','hour','min','dt','zen','dw_solar','dw_solar_QC','uw_solar',
       'uw_solar_QC', 'direct_n','direct_n_QC','diffuse', 'diffuse_QC', 'dw_ir', 'dw_ir_QC', 'dw_casetemp',
       'dw_casetemp_QC', 'dw_dometemp','dw_dometemp_QC','uw_ir', 'uw_ir_QC', 'uw_casetemp','uw_casetemp_QC',
       'uw_dometemp','uw_dometemp_QC','uvb','uvb_QC','par','par_QC','netsolar','netsolar_QC','netir','netir_QC',
       'totalnet','totalnet_QC','temp','temp_QC','rh','rh_QC','windspd','windspd_QC','winddir','winddir_QC',
       'pressure','pressure_QC']


# In[515]:


if run_train:
   # Train Set
   path = r'./data/' + test_location + '/Exp_1_train'
   print("train_path:",path)
   all_files = glob.glob(path + "/*.dat")
   all_files.sort()

   df_big_train = pd.concat([pd.read_csv(f, skipinitialspace = True, quotechar = '"',skiprows=(2),delimiter=' ', 
                    index_col=False,header=None, names=cols) for f in all_files],ignore_index=True)
   print(df_big_train.shape)
   df_train = pd.merge(df_big_train, cs_2010and2011, on=['year','month','day','hour','min'])
   print("loaded training set\n");
   print(df_train.shape)
   


# In[516]:


# Test set
path = r'./data/' + test_location + '/Exp_1_test/' + test_year
print(path)
all_files = glob.glob(path + "/*.dat")
all_files.sort()

df_big_test = pd.concat((pd.read_csv(f, skipinitialspace = True, quotechar = '"',skiprows=(2),delimiter=' ', 
                 index_col=False,header=None, names=cols) for f in all_files),ignore_index=True)
df_test = pd.merge(df_big_test, cs_test, on=['year','month','day','hour','min'])
print('df_test.shape:',df_test.shape)
print("loaded test set\n");
print('df_big_test.shape:',df_big_test.shape)


# ### Merging Clear Sky GHI And the big dataframe

# In[517]:


if run_train:
    # TRAIN set
    #updating the same dataframe by dropping the index columns from clear sky model
    df_train.drop(['index'],axis=1, inplace=True)
    # Resetting Index
    df_train.reset_index(drop=True, inplace=True)


# In[518]:


# TEST set
#updating the same dataframe by dropping the index columns from clear sky model
df_test.drop(['index'],axis=1, inplace=True)
# Resetting Index
df_test.reset_index(drop=True, inplace=True)


# ### Managing missing values

# In[519]:


if run_train:
    # TRAIN set
    #Dropping rows with two or more -9999.9 values in columns
    missing_data_indices = np.where((df_train <=-9999.9).apply(sum, axis=1)>=2)[0] #Get indices of all rows with 2 or more -9999.9
    df_train.drop(missing_data_indices, axis=0, inplace=True) # Drop those inddices
    print('df_train.shape:',df_train.shape)
    df_train.reset_index(drop=True, inplace=True) # 2nd time - Resetting index


# In[520]:


# TEST set
missing_data_indices_test = np.where((df_test <= -9999.9).apply(sum, axis=1)>=2)[0]
df_test.drop(missing_data_indices_test, axis=0, inplace=True)
print('df_test.shape:',df_test.shape)
df_test.reset_index(drop=True, inplace=True) # 2nd time - Reseting Index


# #### First resetting index after dropping rows in the previous part of the code

# In[521]:


if run_train:
    # TRAIN set
    one_miss_train_idx = np.where((df_train <=-9999.9).apply(sum, axis=1)==1)[0]
    print('(len(one_miss_train_idx)',len(one_miss_train_idx))
    df_train.shape

    col_names = df_train.columns
    from collections import defaultdict
    stats = defaultdict(int)
    total_single_missing_values = 0
    for name in col_names:
        col_mean = df_train[~(df_train[name] == -9999.9)][name].mean()
        missing_indices = np.where((df_train[name] == -9999.9))
        stats[name] = len(missing_indices[0])
        df_train[name].loc[missing_indices] = col_mean
        total_single_missing_values += sum(df_train[name] == -9999.9)

    train = np.where((df_train <=-9999.9).apply(sum, axis=1)==1)[0]
    print('len(train):',len(train))


# In[522]:


# TEST set
one_miss_test_idx = np.where((df_test <=-9999.9).apply(sum, axis=1)==1)[0]
len(one_miss_test_idx)
col_names_test = df_test.columns

from collections import defaultdict
stats_test = defaultdict(int)
total_single_missing_values_test = 0
for name in col_names_test:
    col_mean = df_test[~(df_test[name] == -9999.9)][name].mean()
    missing_indices = np.where((df_test[name] == -9999.9))
    stats_test[name] = len(missing_indices[0])
    df_test[name].loc[missing_indices] = col_mean
    total_single_missing_values_test += sum(df_test[name] == -9999.9)
    
test = np.where((df_test <=-9999.9).apply(sum, axis=1)==1)[0]
print('len(test):',len(test))
print('df_test.shape:',df_test.shape);


# ### Exploratory Data analysis
print("Starting exploratory data analysis\n");

# In[523]:


dw_solar_everyday = df_test.groupby(['jday'])['dw_solar'].mean()
ghi_everyday = df_test.groupby(['jday'])['ghi'].mean()
j_day = df_test['jday'].unique()


# In[524]:


fig = plt.figure()

axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
axes1.scatter(j_day,dw_solar_everyday,label='Observed dw_solar',color='red')
axes1.scatter(j_day, ghi_everyday, label='Clear Sky GHI',color='green')

axes1.set_xlabel('Days')
axes1.set_ylabel('Solar Irradiance (Watts /m^2)')
axes1.set_title('Solar Irradiance - Test Year 2009')
axes1.legend(loc='best')

fig.savefig('RNN Paper Results/Exp2_1/' + test_location + '/'+  test_year + 'Figure 2.jpg', bbox_inches = 'tight')


# In[525]:


sns.jointplot(x=dw_solar_everyday,y=ghi_everyday,kind='reg')
plt.xlabel('Observed global downwelling solar (Watts/m^2)')
plt.ylabel('Clear Sky GHI (Watts/m^2)')
plt.savefig('RNN Paper Results/Exp2_1/' + test_location + '/'+  test_year + 'Figure 3', bbox_inches='tight')


# ### making the Kt (clear sky index at time t) column by first removing rows with ghi==0

# In[526]:


if run_train:
    # TRAIN dataset
    df_train = df_train[df_train['ghi']!=0]
    df_train['Kt'] = df_train['dw_solar']/df_train['ghi']
    df_train.reset_index(inplace=True)

    print("train Kt max: "+str(df_train['Kt'].max()))
    print("train Kt min: "+str(df_train['Kt'].min()))
    print("train Kt mean: "+str(df_train['Kt'].mean()))


# In[527]:


# TEST dataset
df_test = df_test[df_test['ghi']!=0]
df_test['Kt'] = df_test['dw_solar']/df_test['ghi']
df_test.reset_index(inplace=True)

print("test Kt max: "+str(df_test['Kt'].max()))
print("test Kt min: "+str(df_test['Kt'].min()))
print("test Kt mean: "+str(df_test['Kt'].mean()))


# In[528]:


if run_train:
    # TRAIN dataset
    df_train= df_train[df_train['Kt']< 5000]
    df_train= df_train[df_train['Kt']> -1000]


# In[529]:


# Test dataset
df_test= df_test[df_test['Kt']< 5000]
df_test= df_test[df_test['Kt']> -1000]


# ### Making 4 Kt columns

# In[530]:


if run_train:
    # Train dataset
    df_train['Kt_2'] = df_train['Kt']
    df_train['Kt_3'] = df_train['Kt']
    df_train['Kt_4'] = df_train['Kt']


# In[531]:


# Test dataset
df_test['Kt_2'] = df_test['Kt']
df_test['Kt_3'] = df_test['Kt']
df_test['Kt_4'] = df_test['Kt']


# #### Group the data (train dataframe)

# In[532]:

if run_train:

    zen = df_train.groupby(['year','month','day','hour'])['zen'].mean()
    dw_solar = df_train.groupby(['year','month','day','hour'])['dw_solar'].mean()
    uw_solar = df_train.groupby(['year','month','day','hour'])['uw_solar'].mean()
    direct_n = df_train.groupby(['year','month','day','hour'])['direct_n'].mean()
    diffuse = df_train.groupby(['year','month','day','hour'])['diffuse'].mean()
    dw_ir = df_train.groupby(['year','month','day','hour'])['dw_ir'].mean()
    dw_casetemp = df_train.groupby(['year','month','day','hour'])['dw_casetemp'].mean()
    dw_dometemp = df_train.groupby(['year','month','day','hour'])['dw_dometemp'].mean()
    uw_ir = df_train.groupby(['year','month','day','hour'])['uw_ir'].mean()
    uw_casetemp = df_train.groupby(['year','month','day','hour'])['uw_casetemp'].mean()
    uw_dometemp = df_train.groupby(['year','month','day','hour'])['uw_dometemp'].mean()
    uvb = df_train.groupby(['year','month','day','hour'])['uvb'].mean()
    par = df_train.groupby(['year','month','day','hour'])['par'].mean()
    netsolar = df_train.groupby(['year','month','day','hour'])['netsolar'].mean()
    netir = df_train.groupby(['year','month','day','hour'])['netir'].mean()
    totalnet = df_train.groupby(['year','month','day','hour'])['totalnet'].mean()
    temp = df_train.groupby(['year','month','day','hour'])['temp'].mean()
    rh = df_train.groupby(['year','month','day','hour'])['rh'].mean()
    windspd = df_train.groupby(['year','month','day','hour'])['windspd'].mean()
    winddir = df_train.groupby(['year','month','day','hour'])['winddir'].mean()
    pressure = df_train.groupby(['year','month','day','hour'])['pressure'].mean()
    ghi = df_train.groupby(['year','month','day','hour'])['ghi'].mean()
    Kt = df_train.groupby(['year','month','day','hour'])['Kt'].mean()
    Kt_2 = df_train.groupby(['year','month','day','hour'])['Kt_2'].mean()
    Kt_3 = df_train.groupby(['year','month','day','hour'])['Kt_3'].mean()
    Kt_4 = df_train.groupby(['year','month','day','hour'])['Kt_4'].mean()


# In[533]:
    df_new_train = pd.concat([zen,dw_solar,uw_solar,direct_n,diffuse,dw_ir,dw_casetemp,dw_dometemp,uw_ir,uw_casetemp,uw_dometemp,
                        uvb,par,netsolar,netir,totalnet,temp,rh,windspd,winddir,pressure,ghi,Kt,Kt_2,Kt_3,Kt_4], axis=1)


# #### Groupdata - test dataframe

# In[534]:


test_zen = df_test.groupby(['month','day','hour'])['zen'].mean()
test_dw_solar = df_test.groupby(['month','day','hour'])['dw_solar'].mean()
test_uw_solar = df_test.groupby(['month','day','hour'])['uw_solar'].mean()
test_direct_n = df_test.groupby(['month','day','hour'])['direct_n'].mean()
test_diffuse = df_test.groupby(['month','day','hour'])['diffuse'].mean()
test_dw_ir = df_test.groupby(['month','day','hour'])['dw_ir'].mean()
test_dw_casetemp = df_test.groupby(['month','day','hour'])['dw_casetemp'].mean()
test_dw_dometemp = df_test.groupby(['month','day','hour'])['dw_dometemp'].mean()
test_uw_ir = df_test.groupby(['month','day','hour'])['uw_ir'].mean()
test_uw_casetemp = df_test.groupby(['month','day','hour'])['uw_casetemp'].mean()
test_uw_dometemp = df_test.groupby(['month','day','hour'])['uw_dometemp'].mean()
test_uvb = df_test.groupby(['month','day','hour'])['uvb'].mean()
test_par = df_test.groupby(['month','day','hour'])['par'].mean()
test_netsolar = df_test.groupby(['month','day','hour'])['netsolar'].mean()
test_netir = df_test.groupby(['month','day','hour'])['netir'].mean()
test_totalnet = df_test.groupby(['month','day','hour'])['totalnet'].mean()
test_temp = df_test.groupby(['month','day','hour'])['temp'].mean()
test_rh = df_test.groupby(['month','day','hour'])['rh'].mean()
test_windspd = df_test.groupby(['month','day','hour'])['windspd'].mean()
test_winddir = df_test.groupby(['month','day','hour'])['winddir'].mean()
test_pressure = df_test.groupby(['month','day','hour'])['pressure'].mean()
test_ghi = df_test.groupby(['month','day','hour'])['ghi'].mean()
test_Kt = df_test.groupby(['month','day','hour'])['Kt'].mean()
test_Kt_2 = df_test.groupby(['month','day','hour'])['Kt_2'].mean()
test_Kt_3 = df_test.groupby(['month','day','hour'])['Kt_3'].mean()
test_Kt_4 = df_test.groupby(['month','day','hour'])['Kt_4'].mean()


# In[535]:


df_new_test = pd.concat([test_zen,test_dw_solar,test_uw_solar,test_direct_n,test_diffuse,test_dw_ir,
                         test_dw_casetemp,test_dw_dometemp,test_uw_ir,test_uw_casetemp,test_uw_dometemp,
                    test_uvb,test_par,test_netsolar,test_netir,test_totalnet,test_temp,test_rh,
                         test_windspd,test_winddir,test_pressure,test_ghi,test_Kt,test_Kt_2,test_Kt_3,test_Kt_4], axis=1)


# In[536]:


#df_new_test.loc[2].xs(17,level='day')


# ### Shifting Kt values to make 1 hour ahead forecast

# #### Train dataset

# In[537]:


if run_train:
    levels_index= []
    for m in df_new_train.index.levels:
        levels_index.append(m)
    for i in levels_index[0]:
        for j in levels_index[1]:
            df_new_train.loc[i].loc[j]['Kt'] = df_new_train.loc[i].loc[j]['Kt'].shift(-1)
            df_new_train.loc[i].loc[j]['Kt_2'] = df_new_train.loc[i].loc[j]['Kt_2'].shift(-2)
            df_new_train.loc[i].loc[j]['Kt_3'] = df_new_train.loc[i].loc[j]['Kt_3'].shift(-3)
            df_new_train.loc[i].loc[j]['Kt_4'] = df_new_train.loc[i].loc[j]['Kt_4'].shift(-4)
    df_new_train = df_new_train[~(df_new_train['Kt_4'].isnull())]


# #### Test dataset

# In[538]:


levels_index2= []
for m in df_new_test.index.levels:
    levels_index2.append(m)


# In[539]:


for i in levels_index2[0]:
    for j in levels_index2[1]:
        df_new_test.loc[i].loc[j]['Kt'] = df_new_test.loc[i].loc[j]['Kt'].shift(-1)
        df_new_test.loc[i].loc[j]['Kt_2'] = df_new_test.loc[i].loc[j]['Kt_2'].shift(-2)
        df_new_test.loc[i].loc[j]['Kt_3'] = df_new_test.loc[i].loc[j]['Kt_3'].shift(-3)
        df_new_test.loc[i].loc[j]['Kt_4'] = df_new_test.loc[i].loc[j]['Kt_4'].shift(-4)


# In[540]:


df_new_test = df_new_test[~(df_new_test['Kt_4'].isnull())]


# ### Normalize train and test dataframe

# In[541]:


if run_train:
    # TRAIN set
    train_norm = (df_new_train - df_new_train.mean()) / (df_new_train.max() - df_new_train.min())
    train_norm.reset_index(inplace=True,drop=True)


# In[542]:


# TEST set
test_norm =  (df_new_test - df_new_test.mean()) / (df_new_test.max() - df_new_test.min())
test_norm.reset_index(inplace=True,drop=True)


# ### Making train and test sets with train_norm and test_norm

# #### finding the gcf (greatest common factor) of train and test dataset's length and chop off the extra rows to make it divisible with the batchsize

# In[543]:


import math
def roundup(x):
    return int(math.ceil(x / 100.0)) * 100


# In[544]:


if run_train:
    # TRAIN set
    train_lim = roundup(train_norm.shape[0])
    train_random = train_norm.sample(train_lim-train_norm.shape[0])
    train_norm = train_norm.append(train_random)

    X1 = train_norm.drop(['Kt','Kt_2','Kt_3','Kt_4'],axis=1)
    y1 = train_norm[['Kt','Kt_2','Kt_3','Kt_4']]

    print("X1_train shape is {}".format(X1.shape))
    print("y1_train shape is {}".format(y1.shape))

    X_train = np.array(X1)
    y_train  = np.array(y1)


# In[545]:


# TEST set
test_lim = roundup(test_norm.shape[0])
test_random = test_norm.sample(test_lim-test_norm.shape[0])
test_norm = test_norm.append(test_random)

X2 = test_norm.drop(['Kt','Kt_2','Kt_3','Kt_4'],axis=1)
y2 = test_norm[['Kt','Kt_2','Kt_3','Kt_4']]

print("X2_test shape is {}".format(X2.shape))
print("y2_test shape is {}".format(y2.shape))

X_test = np.array(X2)
y_test = np.array(y2)


# ### start of RNN

# In[546]:


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable


# In[547]:


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        #Hidden Dimension
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        #Building the RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initializing the hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        #One time step (the last one perhaps?)
        out, hn = self.rnn(x, h0)
        
        # Indexing hidden state of the last time step
        # out.size() --> ??
        #out[:,-1,:] --> is it going to be 100,100
        out = self.fc(out[:,-1,:])
        # out.size() --> 100,1
        return out
        


# In[548]:


if run_train:
    # Instantiating Model Class
    input_dim = 22
    hidden_dim = 15
    layer_dim = 1
    output_dim = 4
    batch_size = 100

    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

    # Instantiating Loss Class
    criterion = nn.MSELoss()

    # Instantiate Optimizer Class
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # initializing lists to store losses over epochs:
    train_loss = []
    train_iter = []
    print("Preparing model to train");
else:
    model = torch.load('RNN Paper Results/Exp2_1/' + test_location + '/torch_model_2010_2011')
    print("Loaded model from file\n");


# In[549]:


# TEst set


test_loss = []
test_iter = []
# converting numpy array to torch tensor

X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

# Convert to Float tensor

X_test = X_test.type(torch.FloatTensor)
y_test = y_test.type(torch.FloatTensor)


# In[550]:


if run_train:
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_train = X_train.type(torch.FloatTensor)
    y_train = y_train.type(torch.FloatTensor)
    # Training the model
    seq_dim = 1

    n_iter =0
    num_samples = len(X_train)
    test_samples = len(X_test)
    batch_size = 100
    num_epochs = 1000
    feat_dim = X_train.shape[1]


    for epoch in range(num_epochs):
        for i in range(0, int(num_samples/batch_size -1)):


            features = Variable(X_train[i*batch_size:(i+1)*batch_size, :]).view(-1, seq_dim, feat_dim)
            Kt_value = Variable(y_train[i*batch_size:(i+1)*batch_size])

            #print("Kt_value={}".format(Kt_value))

            optimizer.zero_grad()

            outputs = model(features)
            #print("outputs ={}".format(outputs))

            loss = criterion(outputs, Kt_value)

            train_loss.append(loss.data[0])
            train_iter.append(n_iter)

            #print("loss = {}".format(loss))
            loss.backward()

            optimizer.step()

            n_iter += 1  
            test_batch_mse =list()    
            if n_iter%100 == 0:
                for i in range(0,int(test_samples/batch_size -1)):
                    features = Variable(X_test[i*batch_size:(i+1)*batch_size, :]).view(-1, seq_dim, feat_dim)
                    Kt_test = Variable(y_test[i*batch_size:(i+1)*batch_size])

                    outputs = model(features)

                    test_batch_mse.append(np.mean([(Kt_test.data.numpy() - outputs.data.numpy().squeeze())**2],axis=1))

                test_iter.append(n_iter)
                test_loss.append(np.mean([test_batch_mse],axis=1))

                print('Epoch: {} Iteration: {}. Train_MSE: {}. Test_MSE: {}'.format(epoch, n_iter, loss.data[0], test_loss[-1]))       
    torch.save(model,'RNN Paper Results/Exp2_1/' + test_location + '/torch_model_2010_2011')


# In[551]:


# JUST TEST CELL

batch_size = 100
seq_dim = 1
test_samples = len(X_test)
batch_size = 100
feat_dim = X_test.shape[1]

# initializing lists to store losses over epochs:
test_loss = []
test_iter = []
test_batch_mse = list()



for i in range(0,int(test_samples/batch_size -1)):
    features = Variable(X_test[i*batch_size:(i+1)*batch_size, :]).view(-1, seq_dim, feat_dim)
    Kt_test = Variable(y_test[i*batch_size:(i+1)*batch_size])
                
    outputs = model(features)
                
    test_batch_mse.append(np.mean([(Kt_test.data.numpy() - outputs.data.numpy().squeeze())**2],axis=1))
                
    test_iter.append(i)
    test_loss.append(np.mean([test_batch_mse],axis=1))


# In[552]:


if run_train:
    print("len(train_loss):",len(train_loss))
    plt.plot(train_loss,'-')


# In[553]:


print("len(test_loss):",len(test_loss))
figLoss = plt.figure()
plt.plot(np.array(test_loss).squeeze(),'r')

figLoss.savefig('RNN Paper Results/Exp2_1/' + test_location + '/'+  test_year + 'test_loss.jpg', bbox_inches = 'tight')


# #### Demornamization

# In[554]:


mse_1 = np.array(test_loss).squeeze()[-1][0]
mse_2 = np.array(test_loss).squeeze()[-1][1]
mse_3 = np.array(test_loss).squeeze()[-1][2]
mse_4 = np.array(test_loss).squeeze()[-1][3]

rmse_1 = np.sqrt(mse_1)
rmse_2 = np.sqrt(mse_2)
rmse_3 = np.sqrt(mse_3)
rmse_4 = np.sqrt(mse_4)

print("rmse_1:",rmse_1)
print("rmse_2:",rmse_2)
print("rmse_3:",rmse_3)
print("rmse_4:",rmse_4)


# In[555]:


rmse_denorm1 = (rmse_1 * (df_new_test['Kt'].max() - df_new_test['Kt'].min()))+ df_new_test['Kt'].mean()
rmse_denorm2 = (rmse_2 * (df_new_test['Kt_2'].max() - df_new_test['Kt_2'].min()))+ df_new_test['Kt_2'].mean()
rmse_denorm3 = (rmse_3 * (df_new_test['Kt_3'].max() - df_new_test['Kt_3'].min()))+ df_new_test['Kt_3'].mean()
rmse_denorm4 = (rmse_4 * (df_new_test['Kt_4'].max() - df_new_test['Kt_4'].min()))+ df_new_test['Kt_4'].mean()

print("rmse_denorm1:",rmse_denorm1)
print("rmse_denorm2:",rmse_denorm2)
print("rmse_denorm3:",rmse_denorm3)
print("rmse_denorm4:",rmse_denorm4)


# In[556]:


rmse_mean = np.mean([rmse_denorm1, rmse_denorm2, rmse_denorm3, rmse_denorm4])
print("rmse_mean:",rmse_mean)


# In[557]:


print(df_new_test['Kt'].describe())
print('\n')
print(df_new_test['Kt_2'].describe())
print('\n')
print(df_new_test['Kt_3'].describe())
print('\n')
print(df_new_test['Kt_4'].describe())


# In[558]:


# Write to file
#f=open('RNN Paper Results/Exp2_1/' + test_location + '/'+  test_year + 'results.txt', "a+")
#f.write(...)


# ### Saving train and test losses to a csv

# In[559]:


if run_train:
    df_trainLoss = pd.DataFrame(data={'Train Loss':train_loss}, columns=['Train Loss'])
    df_trainLoss.head()


# In[560]:


testloss_unsqueezed = np.array(test_loss).squeeze()


# In[561]:


df_testLoss = pd.DataFrame(data=testloss_unsqueezed,columns=['mse1','mse2', 'mse3', 'mse4'])
df_testLoss.head()


# In[562]:


df_testLoss.to_csv('RNN Paper Results/Exp2_1/' + test_location + '/' +  test_year + '_TestLoss.csv')
if run_train:
    df_trainLoss.to_csv('RNN Paper Results/Exp2_1/' + test_location + '/'+  test_year + '_TrainLoss.csv')

