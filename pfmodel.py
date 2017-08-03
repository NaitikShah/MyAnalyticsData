# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:34:46 2017

@author: ecolibrium
"""

from custom_constants import *
from common_utilities import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


my_cm = matplotlib.cm.get_cmap('Spectral')
#for i in range(6,12):
#    df = pd.DataFrame()   
#    df = pd.read_csv('CSVFiles/fiat4140_'+str(i)+'_16.csv',';')
#    
#    df.TimeStamp = pd.to_datetime(df["TimeStamp"])   
##    df.epoch = df.TimeStamp.apply(to_epoch)
##    df.datadate = df.TimeStamp.apply(dateparse)
#    plt.figure()    
#    plt.scatter(df.KW,df.KVA,c=df.TimeStamp.dt.month,linewidth=0,s=5,cmap=my_cm)
#    plt.colorbar()
#    
df = pd.DataFrame()
#for i in range(7,13):   
#    df1 = pd.read_csv('CSVFiles/godrej10944_'+str(i)+'_16.csv',',')
#    df = df.append(df1,ignore_index=False)  
for i in range(2,8):   
    df1 = pd.read_csv('CSVFiles/skf11311_'+str(i)+'_17.csv',',')
    df = df.append(df1,ignore_index=False)
df.TimeStamp = pd.to_datetime(df["TimeStamp"])
 
#Below is calculated KW and KVA when data is not present or its 0(zero)   
df["KVA_cal"] = (df.VLL*df.I*np.sqrt(3))/1000
df["KW_cal"] = (df.VLL*df.I*df.PF*np.sqrt(3))/1000

fig = plt.figure()
ax1 = plt.subplot(121)
ax1.title.set_text('Coloured By PF')
plt.scatter(df.KW,df.KVA,c=df.TimeStamp.dt.month,linewidth=0,s=5,cmap=my_cm)
plt.colorbar()
plt.xlabel("KW",fontsize=14)
plt.ylabel("KVA",fontsize=14)

#fig = plt.figure("Colured By Month")
ax2 = plt.subplot(122, sharex=ax1)
ax2.title.set_text('In ColorBar 2 represents Feb-2017 and 7 represents July-2017')
plt.scatter(df.KW,df.KVA,c=df.RAW_PF,vmax=0.7,vmin=1,linewidth=0,s=5,cmap=my_cm)
plt.colorbar()
plt.xlabel("KW",fontsize=14)
plt.ylabel("KVA",fontsize=14)



fig = plt.figure()
df_new1 = df[((df["TimeStamp"].dt.month == 2) & (df["TimeStamp"].dt.year == 2017))]
df_new2 = df[((df["TimeStamp"].dt.month == 7) & (df["TimeStamp"].dt.year == 2017))]
ax1 = plt.subplot(121)
ax1.title.set_text('February 2017 VS July 2017')
plt.scatter(df_new1.KW,df_new1.KVA,c='b',linewidth=0,s=5)
plt.scatter(df_new2.KW,df_new2.KVA,c='r',linewidth=0,s=5,alpha = 0.1)

plt.xlabel("KW",fontsize=14)
plt.ylabel("KVA",fontsize=14)

df_new1_2 = df[((df["TimeStamp"].dt.month == 2) & (df["TimeStamp"].dt.year == 2017))]
df_new2_2 = df[((df["TimeStamp"].dt.month == 7) & (df["TimeStamp"].dt.year == 2017))]
ax2 = plt.subplot(122, sharex=ax1)
ax2.title.set_text('February 2017 VS July 2017 - Colored by Power Factor')
plt.scatter(df_new1_2.KW,df_new1_2.KVA,c=df_new1_2.RAW_PF,vmax=1,vmin=0.8,linewidth=0,s=5,cmap=my_cm)
plt.scatter(df_new2_2.KW,df_new2_2.KVA,c=df_new2_2.RAW_PF,vmax=1,vmin=0.8,linewidth=0,s=5,cmap=my_cm)
plt.colorbar()
plt.xlabel("KW",fontsize=14)
plt.ylabel("KVA",fontsize=14)
#plt.suptitle("KW VS KVA --- June2016(Blue) VS July2017(Red) --- Right Graph Coloured by PF", fontsize=14)


fig = plt.figure()
ax1 = plt.subplot(121)
ax1.title.set_text('Coloured By PF')
plt.scatter(df.RAW_PF,df.I,c=df.TimeStamp.dt.month,linewidth=0,s=5,cmap=my_cm)
plt.colorbar()
plt.xlabel("RAW_PF",fontsize=14)
plt.ylabel("I",fontsize=14)

#fig = plt.figure("Colured By Month")
ax2 = plt.subplot(122, sharex=ax1)
ax2.title.set_text('In ColorBar 2 represents Feb-2017 and 7 represents July-2017')
plt.scatter(df.RAW_PF,df.I,c=df.RAW_PF,vmax=0.7,vmin=1,linewidth=0,s=5,cmap=my_cm)
plt.colorbar()
plt.xlabel("RAW_PF",fontsize=14)
plt.ylabel("I",fontsize=14)
