
import os
import unittest,random,pdb,pytest

## Datetime import
import datetime

## vtools import.
from vtools.data.timeseries import rts
from vtools.data.vtime import *
from vtools.data.sample_series import *

## numpy testing suite import.
from numpy.testing import assert_array_equal, assert_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal,\
                          assert_array_equal
## numpy import.
import numpy as np
## Local import 
from vtools.functions.filter import *
from pylab import *
from scipy.signal import lfilter,filtfilt
from scipy.signal.filter_design import butter
 

class TestFilter(unittest.TestCase):

    """ test functionality of shift operations """

    def __init__(self,methodName="runTest"):

        super(TestFilter,self).__init__(methodName)
                         
        # Number of data in a time series.
        self.num_ts=1000
        self.max_val=1000
        self.min_val=0.01
        self.large_data_size=100000
        self.two_to_ten = 2**10
        self.test_interval=[minutes(30),hours(2),days(1)]
   
    def setUp(self):
        pass        
        #self.out_file=open("result.txt","a")
                 
    def tearDown(self):
        pass
        #self.out_file.close()    
                
            
    def test_butterworth(self):
        """ Butterworth filter on isa series of 1hour interval with four
            frequencies.
        """
        # Test operations on ts of varied values.
        test_ts=[(pd.Timestamp(1990,2,3,11,15),\
                  self.two_to_ten,hours(1))]

        f1=0.76
        f2=0.44
        f3=0.95
        f4=1.23
        pi=np.pi
        av1=f1*pi/12.
        av2=f2*pi/12.
        av3=f3*pi/12.
        av4=f4*pi/12.
      
        for (st,num,delta) in test_ts:
            ## this day contains components of with frequecies of 0.76/day,
            ## 0.44/day, 0.95/day, 1.23/day            
            data=[np.sin(av1*k)+0.7*np.cos(av2*k)+2.4*np.sin(av3*k) 
                  +0.1*np.sin(av4*k) for k in np.arange(num)] 
                                
            # This ts is the orignial one.           
            ts0=rts(data,st,delta)   
            ts=butterworth(ts0,cutoff_period=hours(40))
            self.assertTrue(ts.index.freq == ts0.index.freq)
            
    def test_butterworth_noevenorder(self):
        """ test a butterworth with non even order input
        """
        start=pd.Timestamp(2000,2,3)
        freq=hours(1)
        data=np.arange(100)
        order=7
        ts0=rts(data,start,freq)
        self.assertRaises(ValueError,butterworth,ts0,order)

        
    def test_godin(self):
        """ test a godin filter on a series of 1hour interval with four
            frequencies.
        """
        # Test operations on ts of varied values.
        test_ts=[(pd.Timestamp(1990,2,3,11, 15),\
                  self.two_to_ten,hours(1))]

        f1=0.76
        f2=0.44
        f3=0.95
        f4=1.23
        av1=f1*np.pi/12
        av2=f2*np.pi/12
        av3=f3*np.pi/12
        av4=f4*np.pi/12
                 
        for (st,num,delta) in test_ts:
            ## this day contains components of with frequecies of 0.76/day,
            ## 0.44/day, 0.95/day, 1.23/day            
            data=[np.sin(av1*k)+0.7*np.cos(av2*k)+2.4*np.sin(av3*k) 
                  +0.1*np.sin(av4*k) for k in range(num)] 
                                
            # This ts is the orignial one.           
            ts0=rts(data,st,delta)   
            ts=godin(ts0)
            self.assertTrue(ts.index.freq == ts.index.freq)
            
    def test_godin_15min(self):        
        """ test godin filtering on a 15min constant values 
            data series with a nan.
        """        
        data=[1.0]*800+[2.0]*400+[1.0]*400
        data=np.array(data)
        data[336]=np.nan
        st=pd.Timestamp(1990,2,3,11, 15)
        delta=minutes(15)
        test_ts=rts(data,st,delta)
        nt3=godin(test_ts)
        npout = nt3.to_numpy().ravel()
        self.assertTrue(np.all(np.isnan(npout[0:144])))
        assert_array_almost_equal(npout[144:192],[1]*48,12)
        self.assertTrue(np.all(np.isnan(npout[192:481])))
        assert_array_almost_equal(npout[481:656],[1.]*175,12)
        self.assertTrue(np.all(np.greater(nt3.to_numpy()[656:944],1)))
        self.assertAlmostEqual(npout[868],1.916618441)
        assert_array_almost_equal(npout[944:1056],[2.]*112,12)
        self.assertTrue(np.all(np.greater(npout[1056:1344],1)))
        self.assertAlmostEqual(npout[1284],1.041451845)
        assert_array_almost_equal(npout[1344:1456],[1.]*112,12)
        self.assertTrue(np.all(np.isnan(npout[1456:1600])))   
        
        nt4 = godin(test_ts.squeeze())
        assert_array_almost_equal(npout[481:656],nt4.to_numpy().ravel()[481:656])        


                                   
    def test_godin_2d(self):
        
        """ Test godin filter on 2-dimensional data set."""
        st=pd.Timestamp(1990,2,3,11, 15)

        ndx = pd.date_range(start = st,freq='15min',periods= 1600)
        d1=[1.0]*800+[2.0]*400+[1.0]*400
        d2=[1.0]*800+[2.0]*400+[1.0]*400
        df = pd.DataFrame({"x":d1,"y":d2},index=ndx)
        df.iloc[336,:]=np.nan
        
        nt3=godin(df)
        d1=nt3.to_numpy()[:,0]
        d2=nt3.to_numpy()[:,1]
        #self.assertTrue(np.all(np.isnan(d1[0:144])))
        assert_array_almost_equal(d1[144:192],[1]*48,12)
        self.assertTrue(np.all(np.isnan(d1[192:481])))
        assert_array_almost_equal(d1[481:656],[1]*175,12)
        self.assertTrue(np.all(np.greater(d1[656:944],1)))
        self.assertAlmostEqual(d1[868],1.916618441)
        assert_array_almost_equal(d1[944:1056],[2]*112,12)
        self.assertTrue(np.all(np.greater(d1[1056:1344],1)))
        self.assertAlmostEqual(d1[1284],1.041451845)
        assert_array_almost_equal(d1[1344:1456],[1]*112,12)
        self.assertTrue(np.all(np.isnan(d1[1456:1600]))) 
        
        self.assertTrue(np.all(np.isnan(d2[0:144])))
        assert_array_almost_equal(d2[144:192],[1]*48,12)
        self.assertTrue(np.all(np.isnan(d2[192:481])))
        assert_array_almost_equal(d2[481:656],[1]*175,12)
        self.assertTrue(np.all(np.greater(d2[656:944],1)))
        self.assertAlmostEqual(d2[868],1.916618441)
        assert_array_almost_equal(d2[944:1056],[2]*112,12)
        self.assertTrue(np.all(np.greater(d2[1056:1344],1)))
        self.assertAlmostEqual(d2[1284],1.041451845)
        assert_array_almost_equal(d2[1344:1456],[1]*112,12)
        self.assertTrue(np.all(np.isnan(d2[1456:1600]))) 
    
    def test_godin_fir(self):
        fir0 = generate_godin_fir('60T')
        fir1 = generate_godin_fir('1H')
        fir2 = generate_godin_fir('1H')
        assert_array_equal(fir0,fir1)
        assert_array_equal(fir0,fir2)
    
    def test_lanczos_cos_filter_coef(self):
        """ Test the sum of lanczos filter coefficients"""    
        cf=0.2
        m=10
        coef=lowpass_cosine_lanczos_filter_coef(cf,m,False)
        coef=np.array(coef)
        coefsum=np.sum(coef)
        self.assertAlmostEqual(np.abs(1.0-coefsum),0.0,places=1)
        
        m=40
        coef=lowpass_cosine_lanczos_filter_coef(cf,m,False)
        coef=np.array(coef)
        coefsum=np.sum(coef)
        self.assertAlmostEqual(np.abs(1.0-coefsum),0.0,places=3)
    
    def test_lanczos_cos_filter_phase_neutral(self):
        """ Test the phase neutriality of cosine lanczos filter"""
        
        ## a signal that is sum of two sine waves with periods of 
        #  4d and 0.25d
        t=np.linspace(0,2000,2001)
        xlow=np.cos(2*np.pi*t/(4.*24.))
        xhigh=np.cos(2*np.pi*t/6.)
        x=xlow+xhigh
        st=pd.Timestamp(1990,2,3,11, 15)
        delta=hours(1)
        ts=rts(x,st,delta)
        ## cutoff period is 30 hours, filterd result should be xlow
        ## approximately
        nt1=cosine_lanczos(ts,cutoff_period=hours(30),filter_len=200)
        nt1.columns = ["nt1"]
        nt1["nt2"] = xlow
        absdiff = (nt1.nt1 - nt1.nt2).abs().max()
        self.assertAlmostEqual(absdiff,0,places=1)
        
    
    def test_lanczos_cos_filter_period_freq_api(self):
        """ Test the cutoff period and frequency of filter"""
        
        ## a signal that is sum of two sine waves with frequency of
        ## 5 and 250HZ, sampled at 2000HZ
        t=np.linspace(0,1.0,2001)
        xlow=np.sin(2*np.pi*5*t)
        xhigh=np.sin(2*np.pi*250*t)
        x=xlow+xhigh
        st=pd.Timestamp(1990,2,3,11, 15)
        delta=hours(1)
        ts=rts(x,st,delta)
        
    
        nt1=cosine_lanczos(ts,cutoff_period=hours(30),filter_len=20,\
                          padtype="even")
        
        ## cutoff_frequency is expressed as ratio of nyquist frequency
        ## ,which is 1/0.5/hours
        cutoff_frequency=2.0/30
        nt2=cosine_lanczos(ts,cutoff_frequency=cutoff_frequency,filter_len=20,\
                           padtype="even")
        
        self.assertEqual(np.abs(nt1.to_numpy()-nt2.to_numpy()).max(),0)
        
    def test_lanczos_cos_filter_len_api(self):
        """ Test the filter len api of the cosine filter"""
        
        ## a signal that is sum of two sine waves with frequency of
        ## 5 and 250HZ, sampled at 2000HZ
        t=np.linspace(0,1.0,2001)
        xlow=np.sin(2*np.pi*5*t)
        xhigh=np.sin(2*np.pi*250*t)
        x=xlow+xhigh
        st=pd.Timestamp(1990,2,3,11, 15)
        delta=hours(1)
        ts=rts(x,st,delta)
        
        ## filter len is none
        nt1=cosine_lanczos(ts,cutoff_period=hours(40),padtype="even")

        self.assertTrue(nt1.index.freq == ts.index.freq)
        ## filter len by defaut lis 40*1.25=50, use it explicitly and
        ## see if the result is the same as the nt1
        nt2=cosine_lanczos(ts,cutoff_period=hours(40),filter_len=50,padtype="even")
        self.assertEqual(np.abs(nt1.to_numpy()-nt2.to_numpy()).max(),0)
   
    def test_lanczos_cos_filter_len(self):
        """ test cosine lanczos input filter length api"""
        
        data=[2.0*np.cos(2*pi*i/5+0.8)+3.0*np.cos(2*pi*i/45+0.1)\
             +7.0*np.cos(2*pi*i/55+0.3) for i in range(1000)]
        data=np.array(data)
        st=pd.Timestamp(1990,2,3,11, 15)
        delta=hours(1)
        ts=rts(data,st,delta)
        
        filter_len=24
        t1=cosine_lanczos(ts,cutoff_period=hours(30),filter_len=filter_len)
        
        filter_len=days(1)
        t2=cosine_lanczos(ts,cutoff_period=hours(30),filter_len=filter_len)
        
        assert_array_equal(t1.to_numpy(),t2.to_numpy())
        
        
        filter_len="invalid"
        self.assertRaises(TypeError,cosine_lanczos,ts,cutoff_period=hours(30),\
                          filter_len=filter_len)
   

    def test_lanczos_cos_filter_nan(self):
        """ Test the data with a nan filtered by cosine lanczos filter"""
        data=[2.0*np.cos(2*pi*i/5+0.8)+3.0*np.cos(2*pi*i/45+0.1)\
             +7.0*np.cos(2*pi*i/55+0.3) for i in range(1000)]
        data=np.array(data)
        nanloc=336
        data[nanloc]=np.nan
        st=pd.Timestamp(1990,2,3,11, 15)
        delta=hours(1)
        ts=rts(data,st,delta)
        m=20
     
        nt1=cosine_lanczos(ts,cutoff_period=hours(30),filter_len=m,padtype="even")
        ## result should have nan from nanidx-2*m+2 to nanidx+2*m-1
        nanidx=np.where(np.isnan(nt1.to_numpy()))[0]
        nanidx_should_be=np.arange(nanloc-2*m,nanloc+2*m+1)
        assert_array_equal(nanidx,nanidx_should_be)
        

    def test_godin(self):
        '''
        Test for godin filter by comparing with vtools answer
        The first timeseries is all zeros except for a single one
        The godin filtering of that series yields the coefficients that are being used
        This is compared with the vtools result based on the same.
        '''
        fname_input = os.path.join(os.path.dirname(__file__),
                                   'test_data/godintest1.csv')
        ts = pd.read_csv(fname_input, parse_dates=True, index_col=0)
        # FIXME: better way to do this on parse?
        ts.index.freq = ts.index.inferred_freq
        tsg = godin(ts)
        fname_expected = os.path.join(os.path.dirname(__file__),
                                      'test_data/godintest-vtools.csv')
        tsg_vtools = pd.read_csv(fname_expected, parse_dates=True, index_col=0)
        tsg_vtools.index.freq = tsg_vtools.index.inferred_freq
        pytest.approx(tsg_vtools['05JAN1990':'15FEB1990'].values,
                      tsg['05JAN1990':'15FEB1990'].values)



    def test_gaussian_filter(self):
       """ Test the nan data with gaussian filter"""
       data=[2.0*np.cos(2*pi*i/5+0.8)+3.0*np.cos(2*pi*i/45+0.1)\
             +7.0*np.cos(2*pi*i/55+0.3) for i in range(1000)]
       data=np.array(data)
       nanloc=336
       data[nanloc]=np.nan
       st=pd.Timestamp(1990,2,3,11, 15)
       delta=hours(1)
       ts=rts(data,st,delta)
       sigma=2
     
       nts=[]
       for order in range(4):
           nts.append(ts_gaussian_filter(ts,sigma,order=order))
  
         
              
if __name__=="__main__":
    
    unittest.main()       

   

    

            

        
        

        
            


        
        

        
        

        

        


        

             



            
        
    
        
        

 

        

        
        

    
                    




        

                 

    
    
