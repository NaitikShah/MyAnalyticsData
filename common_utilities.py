
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 2017

@author: Gokul

Purpose : To Calculate the derived parameters from raw HDF5 file and store back to the same

"""

import pandas as pd
import numpy as np
import datetime as dt
import psycopg2
import os


import smtplib

from tables import *
#from kombu import Connection, Exchange, Queue

# Local Imports
import custom_constants
#from pytablesfdw import Aggregation


# HDF5 Class
class Reading(IsDescription):
    ReadingTypeId = Int16Col()
    TimeStamp = Int32Col()
    ReadingValue = Float32Col()

GMT_to_IST = 19800
DATE_FORMAT = '%Y%m%d'
HDF_DIR = '/home/hdf5/DB/Reading/'

dbConnection = "dbname='smartsense' user='postgres' host='smartsen.se' password='p05+9r35'"

# Connection settings for RabbitMQ
#connection_reading = Connection('amqp://blackcobra:khar905hka5and35h@smartsen.se:5672//')
#connection_agg = Connection('amqp://blackcobra:khar905hka5and35h@server2.smartsen.se:5672//')
reading_exchange = None
readingagg_exchange = None
producer_agg = None
producer_reading = None
# initializing a global list for agg queues
readingagg_queues = []

# Initialize RabbitMQ only if needed, currently not in use
def InitRabbitMQ():
    connection_agg.connect()
    connection_reading.connect()
    reading_exchange = Exchange('ReadingExchange', 'topic', durable=True)
    readingagg_exchange = Exchange('readingAggExchange', 'topic', durable=True)
    producer_agg = connection_agg.Producer(serializer='json')
    producer_reading = connection_reading.Producer(serializer='json')

# Initialize the queue
def InitilizeAggQueue():
    for queue_no in range(1, 15):
        readingagg_queues.append(Queue('readingaggq_' + str(queue_no), readingagg_exchange, 
                    routing_key='reading_key_' + str(queue_no)))

# To convert from float to Date-Time stamp
def dateparse(dates):
    return dt.datetime.fromtimestamp(float(dates))

def dateparse_2(date):
    return dt.datetime.fromtimestamp(float(date)).date()

#Converts python datetime object into epoch for applying of rolling function
def to_epoch(dates):
    if dates == '':
        return None
    else:
        return (dates - dt.datetime(2013,1,1)).total_seconds()

# Return Data for Asset
def GetDataFromRawHDF5(SensorId, StartDate, EndDate):

    filenames = []

    columns = ['ReadingTypeId', 'ReadingValue', 'TimeStamp']
    df = pd.DataFrame(columns=columns)

    NoOfDays = abs((StartDate - EndDate).days)
    QueueId = SensorId // 1000

    date_list = [StartDate + dt.timedelta(days=x) for x in range(0, NoOfDays + 1)]
    fNameFormat = "Reading{0}_{1}.h5"

    for dates in range(0, len(date_list)):
        filenames.append(fNameFormat.format(date_list[dates].strftime(DATE_FORMAT), str(QueueId)))

    for files in range(0, len(filenames)):

        filename = filenames[files]
        print('Loading ' + str(HDF_DIR + filename))

        if os.path.isfile(HDF_DIR + filename):
            print "File found"

            pyStore = pd.HDFStore(HDF_DIR + filename, mode='r')
            dft = pd.DataFrame(pyStore['/s_' + str(SensorId) + '/Reading'])
            df = pd.concat([df, dft])
            pyStore.close()

        else:
            print (str(HDF_DIR + filename) + "File not Found")

    return df


# Returns Pivoted Data
def FormatDataFrame(df):

    ReadingTypeIdDict = custom_constants.ReadingTypeIdDict
    df['ReadingTypeId'] = df['ReadingTypeId'].astype(int)
    df['ReadingTypeId'].replace(ReadingTypeIdDict, inplace=True)
    Table = pd.pivot_table(df, values='ReadingValue', index='TimeStamp', columns='ReadingTypeId')

    return Table


# Function to de-pivot DataFrame
def DePivotDataFrame(df):

    Columns = [custom_constants.ReadingTypeIdDictInv[x] for x in df.columns]
    df.columns = Columns
    df['TimeStamp'] = df.index
    df = pd.melt(df, id_vars=['TimeStamp'], value_vars=Columns)
    df.columns = ['TimeStamp', 'ReadingTypeId', 'ReadingValue']

    return df

# Function to write back the calculated values to Raw HDF5
def WriteBacktoHDF5(df, SensorId):

    fNameFormat = "Reading{0}_{1}.h5"
    QueueId = SensorId // 1000
    df['Temp_Time'] = df.TimeStamp.apply(dateparse_2)
    date_list = df.Temp_Time.unique()

    for Date in date_list:

        filename = fNameFormat.format(Date.strftime(DATE_FORMAT), str(QueueId))

        if os.path.isfile(HDF_DIR + filename):
            print "File found " + HDF_DIR + filename
            df_tmp = df[(df.Temp_Time == Date)]

            with open_file(HDF_DIR + filename, mode="a") as WriteFile:

                WriteTable = WriteFile.root._f_get_child("s_" + str(SensorId))

                for i in df_tmp.index.values:

                    reading = WriteTable.Reading.row
                    reading['TimeStamp'] = df_tmp.TimeStamp[i]
                    reading['ReadingValue'] = df_tmp.ReadingValue[i]
                    reading['ReadingTypeId'] = df_tmp.ReadingTypeId[i]
                    reading.append()

                WriteFile.flush()
        else:
            print (str(HDF_DIR + filename) + "File not Found")

    return True


# Function to write back the calculated values to RabbitMQ Queue
def WriteToReadingQueue(df, SensorId):

    sensor_key = 'reading_key_' + str((SensorId - 1) / 1000)

    for i in df_tmp.index.values:
        msg = {}
        msg['SensorId'] = SensorId
        msg['ReadingTypeId'] = df.ReadingTypeId[i]
        msg['TimeStamp'] = df.TimeStamp[i]
        msg['ReadingValue'] = df.ReadingValue[i]

        producer_reading.publish(msg, exchange=reading_exchange, routing_key=sensor_key)

    return True

# Function to write back the calculated values to RabbitMQ Reading Agg Queue
def WriteToAggFiles(df, SensorId, Grain=15):

    reading_agg_dict_list = []

    for i in df.index.values:
        msg = {}
        msg['SensorId'] = SensorId
        msg['StatsTypeId'] = custom_constants.StatIdDict[df.ReadingTypeId[i]]
        msg['Grain'] = Grain
        msg['ReadingTypeId'] = df.ReadingTypeId[i]
        msg['TimeStamp'] = df.TimeStamp[i]
        msg['ReadingValue'] = df.ReadingValue[i]

        reading_agg_dict_list += [msg]

    # print reading_agg_dict_list
    Aggregation.insert_records('Motor Insight', reading_agg_dict_list)

    return True

def CalculateTHD(df):
    df['VTHD'] = df[['VTHD_R','VTHD_Y','VTHD_B']].max(axis=1)
    phase_code = {'VTHD_R': 1, 'VTHD_Y': 2, 'VTHD_B': 3}
    df['THD_Most_Deviant_Phase_Voltage'] = df[['VTHD_R','VTHD_Y','VTHD_B']].apply(lambda x: np.argmax(x), axis=1).replace(phase_code)
    df['ITHD'] = df[['ITHDR','ITHDY','ITHDB']].max(axis=1)
    phase_code = {'ITHDR': 1, 'ITHDY': 2, 'ITHDB': 3}
    df['THD_Most_Deviant_Phase_Current'] = df[['ITHD_R','ITHD_Y','ITHD_B']].apply(lambda x: np.argmax(x), axis=1).replace(phase_code)
    return df

# To calculate kVA if missing
def Calculate_kVA(df):
    df['kVA_R'] = df.VLN_R * df.IR / 1000.0
    df['kVA_B'] = df.VLN_B * df.IB / 1000.0
    df['kVA_Y'] = df.VLN_Y * df.IY / 1000.0
    return df

def imbalance(var1, var2, var3):
    df_vars = pd.DataFrame([var1, var2, var3])
    vars_mean = df_vars.mean()
    imb = pd.DataFrame([abs(var1 - vars_mean),abs(var2 - vars_mean),abs(var3 - vars_mean)])
    return 100*imb.max()/vars_mean

# Returns calculated imbalances IEEE
def CalculateIEEEImbalance(df, RatedKW, scale_IMB=False):
    # Using IEEE definition for Imbalance
    df['V_Imb_IEEE'] = imbalance(df['VLN_R'], df['VLN_B'], df['VLN_Y'])
    df['I_Imb_IEEE'] = imbalance(df['IR'], df['IB'], df['IY'])
    if scale_IMB:   #Useful in motor
        # Adjusted for loading
        df['I_Imb_IEEE'] = (df['I_Imb_IEEE'] * df['KW']) / RatedKW
    return df

# Returns calculated imbalances IEEE
def CalculateNEEMAImbalance(df, RatedKW, scale_IMB=False):
    # Using IEEE definition for Imbalance
    df['V_Imb_NEEMA'] = imbalance(df['VLL_R'], df['VLL_B'], df['VLL_Y'])
    df['I_Imb_NEEMA'] = imbalance(df['IR'], df['IB'], df['IY'])
    if scale_IMB:   #Useful in motor
        # Adjusted for loading
        df['I_Imb_NEEMA'] = (df['I_Imb_NEEMA'] * df['KW']) / RatedKW
    return df

def ewm_operation(df, col, span = 16):
    "To calculate EMA of a given column. Span depends on the time period required to be used"
    df[col + '_EWM'] = df[col].ewm(span=span).mean()

def time_series_simple_anomaly_remover(df, cols, WINDOW_SIZE=20, percent_criteria = 0.2):
    """Pass the dataframe and an array of column names in the DF on which anomaly removal should be applied
    Example: time_series_simple_anomaly_remover(df, ['KW','KVA','IY'])
    """
    df['TIME_INDEX_CRITERIA'] = np.nan
    PERCENTAGE_VARIATION_CRITERIA = percent_criteria
    for col in cols:
        #Left window Moving average
        left_mov_avg = df[col].rolling(window=WINDOW_SIZE).mean().shift(-1 * WINDOW_SIZE)
        left_std = df[col].rolling(window=WINDOW_SIZE).std().shift(-1 * WINDOW_SIZE)
        #Right window moving average
        right_mov_avg = df[col].rolling(window=WINDOW_SIZE).mean().shift(WINDOW_SIZE)
        right_std = df[col].rolling(window=WINDOW_SIZE).std().shift(WINDOW_SIZE)
        std_dev = df[col].std()
        
        #Approve point if its in range of either left or right window moving average
        df['TIME_INDEX_CRITERIA'] = (abs(left_mov_avg - df[col]) <= PERCENTAGE_VARIATION_CRITERIA * left_mov_avg) | (abs(right_mov_avg - df[col]) <= PERCENTAGE_VARIATION_CRITERIA*right_mov_avg)
        print "Points removed: ", df[~df['TIME_INDEX_CRITERIA']][col].shape[0]
        #print "Points removed: ", df[~df['TIME_INDEX_CRITERIA']][col]
        #Drop where criteria is false
        #df = df[df['TIME_INDEX_CRITERIA']] #Used previously, issue was the setting data on copy warning
        df = df.drop(df[~df['TIME_INDEX_CRITERIA']].index)
    del df['TIME_INDEX_CRITERIA']
    return df

# Calculate PoweFactor using KW and KVA
def CalculatePowerFactor(df):

    df['PF_R_Cal'] = df.kW_R / df.kVA_R
    df['PF_B_Cal'] = df.kW_B / df.kVA_B
    df['PF_Y_Cal'] = df.kW_Y / df.kVA_Y

    df = df[np.isfinite(df.PF_R_Cal)]
    df = df[np.isfinite(df.PF_B_Cal)]
    df = df[np.isfinite(df.PF_Y_Cal)]

    return df

def ReadFromDB(query):
    conn = psycopg2.connect(dbConnection)
    result = pd.read_sql_query(query, conn)
    conn.close()
    return result

def WriteToDB(query):
    conn = psycopg2.connect(dbConnection)
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()


"""
    For aggregation of data. Specify time period (optional) for custom period. Be careful using time period, should not be less than data collection period which is typically 1  minute.
Usage Examples:
1) To do 10 mins aggregation without any Std deviation columns, use -- aggregate_param_based(df, '10T') --
2) To do 5 mins aggregation with STD deviation, use  --  aggregate_param_based(df, '5T', True) --
"""
def aggregate_param_based_generic(df, time_period = '15T', add_SD = False):
    interpolated_set = pd.DataFrame(df.index, index=df.index).resample(time_period).sum()
    new_df = pd.concat([interpolated_set, df]).sort_index().interpolate(method='time')
    agg_keys = set(df.columns).intersection(custom_constants.AGG_DICT)
    agg_dict = { k:custom_constants.AGG_DICT[k] for k in agg_keys }
    final_df = new_df.resample(time_period).agg(agg_dict)
    if not add_SD:
        return final_df.dropna()
    else:
        final_df = final_df.dropna()
        stddev_df = new_df.resample(time_period).agg(np.std)
        stddev_df = stddev_df[final_df.columns]
        count_df = pd.DataFrame({'Count': new_df[new_df.columns[0]].resample(time_period).agg(np.count_nonzero) - 2}, index = stddev_df.index).dropna()
        stddev_df.columns = stddev_df.columns.values + '_STDDEV'
        stddev_df = stddev_df.dropna()
        stats_df = pd.concat([stddev_df, count_df], axis = 1)
        return pd.merge(final_df, stats_df, how='inner', left_index=True, right_index=True)
