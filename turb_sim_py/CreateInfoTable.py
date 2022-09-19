# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 08:00:52 2022

@author: victoria.lockridge

Create a table with the needed information for the Purdue code
Fields:
    real image filename
    date created
    cn2 measured
    pixels
    object size
    
    baseline image filename
    
    
    luFolder = "C:\Data\JSSAP";
fn1 = "combined_sharpest_images.xlsx";  % Contains real image names with dates and times
% Match the above file to Cn2 values in the below 2 files
fn0929 = "20210929_combined_atmospherics_600-50-1000.csv"; % Contains cn2 values for date/time/range
fn0930 = "20210930_combined_atmospherics_600-50-1000.csv";
"""

import os
import pandas as pd

dirn = r"C:\Data\JSSAP"
filen = 'combined_sharpest_images.xlsx'
fn0929 = '20210929_combined_atmospherics_600-50-1000.csv' # % Contains cn2 values for date/time/range
fn0930 = '20210930_combined_atmospherics_600-50-1000.csv'
outfile = 'combined_sharpest_images_withCn2.csv'

combinedPath = os.path.join(dirn, filen)
path29 = os.path.join(dirn, fn0929)
path30 = os.path.join(dirn, fn0930)
pathout = os.path.join(dirn, outfile)

def CreateInfoDf():

    colnames = ['date','time','timeSecs', 'range', 'zoom', 'focus', 'imageFn', 'imageHt', 'imageWd', 'pixelStep', 'start','stop']
    dfcomb = pd.read_excel(combinedPath, sheet_name = 'combined_sharpest_images')
    dfcomb.columns = colnames
    dfcomb['obj_size'] = abs(dfcomb['start']) + dfcomb['stop']
    
    #
    colnames29 = ['timeSecs', 'range', 'temp', 'humidity', 'windSpeed', 'windDir', 'barPressure', 'solarLoad', 'cn2'] 
    df29 = pd.read_csv(path29, header = None, skiprows = 1)
    df29.columns = colnames29
    df29.sort_values(['range','timeSecs'], inplace = True)
    df30 = pd.read_csv(path30, header = None, skiprows = 1)
    df30.columns = colnames29
    df30.sort_values(['range','timeSecs'], inplace = True)
    
    #Add cn2 values to dfcomb
    # match date and timeSecs to df name's date and then timeSecs in df with range value
    # find closest timeSecs (before or after)
    # pull cn2 value
    # write value to dfcomb row
    
    dateLU = {'29': df29, '30':df30}
    dfcomb['cn2'] = 0.0
    
    for row in range(len(dfcomb)):
        fdate = str(dfcomb.loc[row,'date'])
        fdateNum = fdate.split('2021-09-')[1][0:2]
        dfdate = dateLU[fdateNum]
        rng = dfcomb.loc[row,'range']
        timeS = dfcomb.loc[row,'timeSecs']
        
        maskR = (dfdate.range == rng)
        dfview = dfdate.loc[maskR,:].copy()
        dfview.reset_index(inplace = True)
        
        dfview['tDiff'] = abs(dfview['timeSecs'] - timeS)
        indmin = dfview.tDiff.argmin()
        cn2 = dfview.loc[indmin, 'cn2']
        
        #  Assign cn2 to dfcomb
        dfcomb.loc[row,'cn2'] = cn2
        
    #dfcomb.to_csv(pathout, index = False)
    return dfcomb
    
