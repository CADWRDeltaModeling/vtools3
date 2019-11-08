""" Tests for downloading scripts
"""
import os
import pandas as pd
from vtools.datastore.download_nwis import nwis_download
from vtools.datastore.download_noaa import noaa_download
from vtools.datastore.download_wdl import wdl_download
from vtools.datastore.download_cdec import cdec_download


def clean_up(expected_artifacts):
    # Clean up
    for f in expected_artifacts:
        if os.path.exists(f):
            os.remove(f)


def test_nwis_download():
    # Setting up
    stations = ['11313452', ]
    dest_dir = '.'
    start = pd.to_datetime('2019-01-01')
    end = pd.to_datetime('2019-03-10')
    # Run tests
    nwis_download(stations, dest_dir, start, end=end, param=None,
                  overwrite=True)
    expected_artifacts = ['{}.rdb'.format(s) for s in stations]
    for fpath in expected_artifacts:
        ts = pd.read_csv(fpath, comment='#', delimiter='\t')
        assert(float(ts.loc[1]['15416_00060']) == -11100.)
    clean_up(expected_artifacts)


def test_wdl_download():
    stations = ['B95060', ]
    years = [2018, ]
    dest_dir = '.'
    wdl_download(stations, years, dest_dir, overwrite=False)
    expected_artifacts = ['{}_gageheight.csv'.format(s) for s in stations]
    for fpath in expected_artifacts:
        ts = pd.read_csv(fpath, header=None)
        assert(ts.loc[0][1] == 5.65)
    clean_up(expected_artifacts)


def test_noaa_download():
    # Setting up
    stations = ['9414290', ]
    dest_dir = '.'
    product = 'water_level'
    start = pd.to_datetime('2019-01-01')
    end = pd.to_datetime('2019-03-10')
    # Run tests
    noaa_download(stations, product, start, end)
    expected_artifacts = ['{}_{}.txt'.format(s, product) for s in stations]
    for fpath in expected_artifacts:
        ts = pd.read_csv(fpath, comment='#')
        assert(ts.loc[0][1] == 0.623)
    clean_up(expected_artifacts)


def test_cdec_download():
    stations = ['OSJ', ]
    dest_dir = '.'
    start = pd.to_datetime('2019-01-01')
    end = pd.to_datetime('2019-03-10')
    param = 'ec'
    cdec_download(stations, dest_dir, start, end=end, param=param,
                  overwrite=False)
    expected_artifacts = ['{}_{}.csv'.format(s, param) for s in stations]
    for fpath in expected_artifacts:
        ts = pd.read_csv(fpath)
        assert(float(ts.loc[0][6]) == 651.)
    clean_up(expected_artifacts)
