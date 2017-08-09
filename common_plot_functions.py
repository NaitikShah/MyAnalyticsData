

#IMPORT THIS FILE FOR PLOTTING AND OTHER UTILITIES
import pandas as pd
import datetime as dt
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import numpy as np
import custom_constants, time
from scipy import stats, signal
from Python import negative_sequence
from mpl_toolkits.mplot3d import Axes3D
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

def dateparse(dates):
    if dates=='':
        return None
    else:
        try:
            #Datetime object from Epoch TS
            return dt.datetime.fromtimestamp(float(dates))
        except:
            #Converts from string to python datetime object            
            return dt.datetime.strptime(str(dates), '%d-%m-%Y %H:%M')

#Plot with X and Y grid lines. Can pass Y as an array of multiple columns. Eg: df.lag, df[['KVA','kVA (B PHASE)']]
def plot_with_grid(x, y):
    #With seaborn - discontinue seaborn as it makes graph colors black and white
    #import seaborn
    #seaborn.set_style('whitegrid')
    plt.plot(x, y)
    plt.show()

def plot_scatter(x, y):
    #With seaborn - discontinue seaborn as it makes graph colors black and white
    #import seaborn
    #seaborn.set_style('whitegrid')
    plt.scatter(x, y)
    plt.show()

#2 timeseries with different interval on same plot

#Pass timeseries and their respective params. Eg: To plot HT side V, I, PF and LT side V, PF on same timeline, pass as [htdata.V, htdata.I, htdata.PF, ltdata.V, ltdata.PF]
def plotTS(dfs, styles=['bo', 'go','ro', 'co', 'mo', 'yo' ], markersize=2, linestyle='-'):
    from matplotlib.dates import date2num, DateFormatter
    fig, ax = plt.subplots()
    count = 0
    for df in dfs:
        ax.plot_date(df.index, df, styles[count], markersize=markersize,linestyle=linestyle)
        count = count + 1
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    plt.legend(loc='best')
    plt.show()

#Pass an array of series as 1st argument to plot their histograms in a single plot 
def plotHistogram(dfs, num_bins = 200, start_xlim = None, end_xlim = None, alpha=0.6):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    count = 0
    for df in dfs:
        ax1.hist(df, num_bins, color = colors[count], alpha = alpha)
        count = count + 1
    if start_xlim is not None:
        plt.xlim(start_xlim, end_xlim)
    plt.show()

# Pass X series and Y1 and Y2 series
def plotDualYAxis(x, y1, y2, styles=['g-','b-']):
    #Plot with 2 axes
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, y1, styles[0])
    ax2.plot(x, y2, styles[1])
    plt.show()

# Plots two variable on 1Y axis and the remaining on the other
def plotDualYAxis_3var(x, y1, y2,y3, styles=['g-','b-','r-']):
   #Plot with 2 axes
   fig, ax1 = plt.subplots()
   #fig, ax2 = plt.subplots()
   ax2 = ax1.twinx()
   ax3 = ax1.twinx()
   ax1.plot(x, y1, styles[0])
   ax2.plot(x, y2, styles[1])
   ax3.plot(x, y3, styles[2])
   plt.show()
   
#Plot correlation matrix of all columns within the dataframe
def plot_corr(df, method = 'pearson'):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.
    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    corr = df.corr(method = method)
    size = df.columns.size
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns,  rotation='vertical');
    plt.yticks(range(len(corr.columns)), corr.columns);
    plt.show()


def plot_colorby_criteria(df_x, df_y, df_criteria, criteria_name='color_criteria'):
    df_x = pd.Series(df_x) #as .name does not work on npseries
    df_y = pd.Series(df_y)
    fig = plt.scatter(df_x, df_y, c = df_criteria,s=10,lw=0)
    plt.colorbar(fig).set_label(criteria_name)
    plt.xlabel(df_x.name)
    plt.ylabel(df_y.name)
    plt.title(str(df_y.name) + ' VS ' + str(df_x.name))
    plt.show()  
    
def plotWoW(df, x_col, y_col, alpha = 0.6):
    '''Plots based on assumption that dataframe has a datetime index.
       Usage example: plotWoW(df, 'KVA', 'PF')     
    '''
    count = 0
    if not 'week' in df.columns:
        df['week'] = df.index.week
    df_x = df[x_col]
    df_y = df[y_col]
    fig = plt.scatter(df_x, df_y, c = df['week'])
    plt.colorbar(fig).set_label('Week')
    plt.xlabel(df_x.name)
    plt.ylabel(df_y.name)
    plt.title(str(df_x.name) + ' VS ' + str(df_y.name))
    plt.show()

def plotMoM(df, x_col, y_col, alpha = 0.6):
    '''Plots based on assumption that dataframe has a datetime index.
       Usage example: plotMoM(df.index, 'KVA', 'PF')     
    '''
    count = 0
    if not 'month' in df.columns:
        df['monthnames'] = df.index.strftime('%B')
        df['month'] = df.index.month
    df_x = df[x_col]
    df_y = df[y_col]
    fig = plt.scatter(df_x, df_y, c = df['month'])
    cbar = plt.colorbar(fig)
    cbar.set_ticks(df['month'].unique())
    cbar.set_ticklabels(df['monthnames'].unique())
    cbar.set_label = 'Month'
    plt.xlabel(df_x.name)
    plt.ylabel(df_y.name)
    plt.title(str(df_x.name) + ' VS ' + str(df_y.name))
    plt.show()

#Accepts data frame, creates a CSV and can be used to plot in DyGraphs
def create_dygraph_plot(Data):
    import os
    #Data_Location = os.path.dirname(os.path.realpath(__file__)) + '/DyGraph_plot_Django_app/plotting/static/'
    Data_Location = '/home/eco/smartsense/EI/DyGraph_plot_Django_app/plotting/static/'
    print 'Dumping data to CSV at path:', Data_Location
    Data.to_csv(Data_Location + 'Data.csv', sep = ',', header = True)


"""Remove data points with variance in a time interval to target only those points with uniform loading.
Time interval is in seconds
NOTE: This method requires pandas 0.19 or above
"""
def remove_variance(df, time_interval = 120, std_level = 5.0, params = ['KW_HT']):
    for param in params:
        k = pd.DataFrame()
        std = std_level/100.0
        k['std_dev'] = (df[param].resample(str(time_interval) + 'S').std()) / (df[param].resample(str(time_interval) + 'S').mean())
        k['bad'] = k['std_dev'] > std
        #print "Rows with high variance::: ", k[k['bad'] == True].count()
        df['to_delete'] = False
        for ele in k[k['bad'] == True].index:
            df.ix[ele: ele + timedelta(seconds=time_interval), 'to_delete'] = True
        df.drop(df[df['to_delete'] == True].index, inplace=True)

"""Reads all the csv's in a folder and merges them in a single dataframe with sorted index
""" 
def Read_multiple_csv_to_dataframe(path):   
    import os, glob
    allFiles = glob.glob(os.path.join(path, "*.csv"))
    df = pd.concat((pd.read_csv(f, index_col=[0],date_parser = dateparse, parse_dates=['TimeStamp'])) for f in allFiles)
    df.sort(inplace=True)
    return df

# Daywise description of parameters
"""
Returns a pivot table for all the negative sequence parameters.
"""
def daywise_description(df,ColumnList):
    Temp = pd.DataFrame(columns=['Item','Date'] + ColumnList)
    """ Pass the dataframe and an array of column names for which the pivot table is to be created
        Example: daywise_description(df, ['In', 'Vn', 'Zn'])
    """
    for i in np.unique(df.index.dayofyear):
        obj = df.loc[df.index.dayofyear == i,ColumnList].describe()
        obj['Item'] = obj.index
        obj.index = np.linspace(Temp.index.size , Temp.index.size + 7, num=8) 
        obj['Date'] = np.repeat(df.index[df.index.dayofyear == i].min(),8)
        Temp = pd.concat([Temp,obj])
        
    Temp = Temp.pivot_table(index = 'Date', columns = 'Item', values = ColumnList)
    Temp.columns = [' '.join(col).strip().replace(' ', '') for col in Temp.columns.values] 
    return Temp

def ewm_test(start, end, alpha=1.0/3.0):
    """To check how Exponential avg is calculated, only for reference, not for use"""
    end = end - 1
    sum1 = 0 
    sum2 = 0 
    factor = 1.0 - alpha
    for a in range(start, end, -1):
        sum1 = sum1 + pow(factor, float(start-a))
        sum2 = sum2 + a*pow(factor, float(start-a))
    print sum2/sum1 

def plot_3d(x,y,z):
    fig = plt.figure(); ax = Axes3D(fig); ax.scatter3D(x, y,z)
    plt.show()

"""
Takes a multi-column dataframe as input and performs DBSCAN on it.
Example:
X = df[['KW_LT', 'V_Ratio']]
run_dbscan(X)
"""
def run_dbscan(X, epsilon = 0.09, min_samples = 20):
    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets.samples_generator import make_blobs
    Y = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(Y)
    return db
    ### To plot results after DBSCAN, use as below
    #plot_colorby_criteria(Y[:, 1], Y[:, 0], db.labels_)
