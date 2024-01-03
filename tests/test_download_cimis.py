import pytest
import pandas as pd
from dms_datastore import download_cimis

def test_download_davis():
    davisdf=download_cimis.fetch_data(6)
    print(davisdf.head())
    assert(davisdf.size>=364)

def test_station_list():
    slist=download_cimis.fetch_station_list()
    davis=slist[slist.Name.str.contains('Davis')]
    print(davis)
    assert(len(davis.index)==1)
    assert(int(davis.iloc[0]['Station Number'])==6)

def test_download_davis_2018():
    davisdf=download_cimis.fetch_data_for_year(6,2018)
    print(davisdf.head())
    assert(davisdf.size>=364)

def test_fetch_columnnames():
    colnames=download_cimis.fetch_column_names_from_readme()
    print('Column Names:',colnames)
    assert(len(colnames)>30)
