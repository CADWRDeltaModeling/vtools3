'''
CIMIS provides ETo associated information which is needed for the consumptive use model calculations.
'''
import pandas as pd
import numpy as np
# For dealing with zipped files
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

def fetch_data(sid):
    '''
    Retrieves daily data from FTP site. This data goes back only 1 year and 
    should be used to retrieve recent data as a dataframe.

    sid is the integer station id (it will be zero padded to size 3 to get station id)
    '''
    url='ftp://ftpcimis.water.ca.gov/pub2/daily/DayYrETo%03d.csv'%int(sid)
    df=pd.read_csv(url, header=None, index_col=1, parse_dates=True, dtype={0:'int',2:'float'})
    df=df.drop(axis=1,columns=[0,2,4])
    df.columns=[sid]
    return df

def fetch_column_names_from_readme():
    '''
    Attempts to read column names from readme*.txt file
    '''
    df=pd.read_csv('ftp://ftpcimis.water.ca.gov/pub2/readme-ftp-Revised5units.txt',header=None,skiprows=61,nrows=31,encoding='cp1252')
    return df.iloc[:,0].str.split('\t',n=1,expand=True).iloc[:,1].values

def fetch_data_for_year(sid,year,colnames=[]):
    '''
    Retrieves daily data from FTP site for a year. Only works for past years. Use fetch_data for
    recent data.

    A readme-ftp-*.txt explains the column names. If colnames is None, this readme file is fetched
    and the column names are extracted from it. One can also call and cache those names by using
    function fetch_column_names_from_readme()
    '''
    if len(colnames) < 1:
        colnames=fetch_column_names_from_readme()
    z=urlopen('ftp://ftpcimis.water.ca.gov/pub2/annual/dailyStns%04d.zip'%year)
    myzip=ZipFile(BytesIO(z.read())).extract('%04ddaily%03d.csv'%(year,sid))
    df=pd.read_csv(myzip,parse_dates=[1])
    df.columns=colnames
    return df

def fetch_station_list():
    '''
    Retrieves station list from the FTP site.
    '''
    # get stations list
    stations_url="ftp://ftpcimis.water.ca.gov/pub2/CIMIS%20Stations%20List%20(April18).xlsx"
    slist=pd.read_excel(stations_url,dtype={'Station Number':'str'},parse_dates=['Connect','Disconnect'])
    slist=slist.dropna()
    return slist

