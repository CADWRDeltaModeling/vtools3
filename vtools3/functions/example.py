
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

def gap_size(ts):
    from itertools import groupby
    
    """Find the size of gaps (blocks of nans) for every point in an array.
    Parameters
    ----------
    x : array_like
    an array that possibly has nans

    Returns
    gaps : array_like
    An array the values of which represent the size of the gap (number of nans) for each point in x, which will be zero for non-nan points.    
    """
    
    
    isgap = zeros_like(x)
    isgap[isnan(x)] = 1
    gaps = []
    for a, b in groupby(isgap, lambda x: x == 0):
        if a: # Where the value is 0, simply append to the list
            gaps.extend(list(b))
        else: # Where the value is one, replace 1 with the number of sequential 1's
            l = len(list(b))
            gaps.extend([l]*l)
    return array(gaps)
    
    
    



def main():
    tndx = pd.date_range(start="2019-01-01",end="2019-01-10",freq="H")    
    tnum = np.arange(0.,len(tndx))
    signal = np.cos(tnum*2.*np.pi/24.)

    signal[80:85] = np.nan
    signal[160:168:2] = np.nan
    df = pd.DataFrame({"signal":signal},index=tndx)
    orig = df.index[0]
    x0 = (df.index - orig).total_seconds()
    y0 = df.values
    
    # Express the destination times as a dataframe and append to the source
    tndx2 = pd.DatetimeIndex(['2019-01-04 00:00','2019-01-04 10:17','2019-01-07 16:00'])
    x1 = (tndx2 - orig).total_seconds()
    
    # Extract at destination locations
    good = ~np.isnan(y0).flatten()
    #print x0[good]
    #print y0[good]
    interpolator = scipy.interpolate.interp1d(x0[good], y0[good], kind='cubic', axis=0, fill_value=np.nan, assume_sorted=False)
    interpolated = interpolator(x1)
    #print interpolated



def main1():
    tndx = pd.date_range(start="2019-01-01",end="2019-01-10",freq="H")    
    tnum = np.arange(0.,len(tndx))
    signal = np.cos(tnum*2.*np.pi/24.)

    signal[80:85] = np.nan
    signal[160:168:2] = np.nan
    df = pd.DataFrame({"signal":signal},index=tndx)

    # Express the destination times as a dataframe and append to the source
    tndx2 = pd.DatetimeIndex(['2019-01-04 00:00','2019-01-04 10:17','2019-01-07 16:00'])
    df2 = pd.DataFrame( {"signal": [np.nan,np.nan,np.nan]} , index = tndx2)
    big_df = df.append(df2,sort=True)  
    
    # At this point there are duplicates with NaN values at the bottom of the DataFrame
    # representing the destination points. If these are surrounded by lots of NaNs in the source frame
    # and we want the limit argument to work in the call to interpolate, the frame has to be sorted and duplicates removed.     
    big_df = big_df.loc[~big_df.index.duplicated(keep='first')].sort_index(axis=0,level=0)
    
    # Extract at destination locations
    interpolated = big_df.interpolate(method='cubic',limit=3).loc[tndx2]
    #print interpolated


def main2():
    tndx = pd.date_range(start="2019-01-01",end="2019-01-10",freq="H")    
    tnum = np.arange(0.,len(tndx))
    signal = np.cos(tnum*2.*np.pi/24.)

    signal[80:85] = np.nan
    signal[160:168:2] = np.nan
    
    df = pd.DataFrame({"signal":signal},index=tndx)

    df1= df.resample('15min').interpolate('cubic',limit=9)


    tndx2 = pd.DatetimeIndex(['2019-01-04 00:00','2019-01-04 10:17','2019-01-07 16:00'])
    df2 = pd.DataFrame( {"signal": [np.nan,np.nan,np.nan]} , index = tndx2)
    
    big_df = df.append(df2,sort=True)
    big_df.to_csv("out.csv")

    big_df3 = big_df.copy()
    big_df3["dup"] = big_df.index.duplicated(keep='first')
    big_df3.to_csv("out3.csv")

    big_df2 = big_df.loc[~big_df.index.duplicated(keep='first')].sort_index(axis=0,level=0)
    big_df2.to_csv("out2.csv")
    
    big_df2 = big_df2.interpolate(method='cubic',limit=3)
    
    interpolated = big_df2.loc[tndx2]

    print "[1]"
    print df.iloc[78:90,:]
    print "[2]"
    print big_df2.iloc[78:90,:]
    print "[3]"
    print big_df2.tail(6)
    print "[4]"    
    print interpolated
    
    print "[5]"
    print tndx2
    big_df2.plot()
    plt.show()
    
    
    
if __name__ == "__main__":
    import timeit 
    timeit.timeit('main()',number=100)
    #timeit.timeit('main()',number=100)
    
    
    
    
    