# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 16:10:32 2022

@author: victoria.lockridge
"""

"""
Use combined_sharpest_images.xlsx as the main file.
Find information in 2 below files to enter Temperature, Humidity, Wind spped, Wind Direction, 
Barometric pressure, solar load, and cn2.  After cn2 is entered into 
combined_sharpest_images.xlsx, use range and cn2 to calculate r0.

Files:  20210929_combined_atmospherics_600-50-1000.csv and 
20210930_combined_atmospherics_600-50-1000.csv


"""


import os
import pandas as pd
import numpy as np

def CalculateR0(cn2, L, wvl):
    k = 2*np.pi/wvl
    b0 = 0.158625
    r_inside = b0*k**2*cn2*L
    r0 = np.power(r_inside,-0.6)   # or -3/5
    return r0

dirn = r"C:\Data\JSSAP"
filen = 'combined_sharpest_images.xlsx'
fn0929 = '20210929_combined_atmospherics_600-50-1000.csv' # % Contains cn2 values for date/time/range
fn0930 = '20210930_combined_atmospherics_600-50-1000.csv'
outfile = 'combined_sharpest_images_withAtmos.xlsx'

combinedPath = os.path.join(dirn, filen)
path0929 = os.path.join(dirn, fn0929)
path0930 = os.path.join(dirn, fn0930)
pathout = os.path.join(dirn, outfile)

wvl = 0.525e-6 

if __name__ == "__main__":
    # Read spreadsheets into dataframes
    colnames = ['date','time','timeSecs', 'range', 'zoom', 'focus', 'imageFn', 'imageHt', 'imageWd', 'pixelStep', 'start','stop']
    dfcombo = pd.read_excel(combinedPath, sheet_name = 'combined_sharpest_images')
    dfcombo.columns = colnames
    dfcombo['obj_size'] = abs(dfcombo['start']) + dfcombo['stop']
        
    colnames29 = ['timeSecs', 'range', 'temp', 'humidity', 'windSpeed', 'windDir', 'barPressure', 'solarLoad', 'cn2'] 
    df29 = pd.read_csv(path0929, header = None, skiprows = 1)
    df29.columns = colnames29
    df29.sort_values(['range','timeSecs'], inplace = True)
    df30 = pd.read_csv(path0930, header = None, skiprows = 1)
    df30.columns = colnames29  # Same column names in both files, Sept 29 and Sept 30
    df30.sort_values(['range','timeSecs'], inplace = True)
    
    # Add atmospheric values to dfcombo
    # Match date and timeSecs of atmospheric files to dfcombo, row by row,
    #  using date and then timeSecs within dfcombo row's range value.
    # Find closest timeSecs (before or after).
    # Pull atmospheric values and write to dfcomb row.
    
    dateLookUp = {'29': df29, '30':df30}
    dfcombo['temp'] = 0.0   
    dfcombo['humidity'] = 0.0
    dfcombo['windSpeed'] = 0.0
    dfcombo['windDir'] = 0.0
    dfcombo['barPressure'] = 0.0
    dfcombo['solarLoad'] = 0.0
    dfcombo['cn2'] = 0.0
    dfcombo['r0'] = 0.0
    
    
    for row in range(len(dfcombo)):
        # Parse date to get write atmospheric file/dataframe
        fdate = str(dfcombo.loc[row,'date'])
        fdateNum = fdate.split('2021-09-')[1][0:2]
        dfdate = dateLookUp[fdateNum]
        rng = dfcombo.loc[row,'range']
        timeS = dfcombo.loc[row,'timeSecs']
        
        # Pull data by range value
        maskR = (dfdate.range == rng)
        dfview = dfdate.loc[maskR,:].copy()
        dfview.reset_index(inplace = True)
        
        # Find closest time using time in seconds
        dfview['tDiff'] = abs(dfview['timeSecs'] - timeS)
        indmin = dfview.tDiff.argmin()
        
        #  Assign atmospheric values to dfcombo        
        dfcombo.loc[row,'temp'] = dfview.loc[indmin, 'temp']
        dfcombo.loc[row,'humidity'] = dfview.loc[indmin, 'humidity']
        dfcombo.loc[row,'windSpeed'] = dfview.loc[indmin, 'windSpeed']
        dfcombo.loc[row,'windDir'] = dfview.loc[indmin, 'windDir']
        dfcombo.loc[row,'barPressure'] = dfview.loc[indmin, 'barPressure']
        dfcombo.loc[row,'solarLoad'] = dfview.loc[indmin, 'solarLoad']
        dfcombo.loc[row,'cn2'] = dfview.loc[indmin, 'cn2']
        dfcombo.loc[row,'r0'] = CalculateR0(dfcombo.loc[row,'cn2'], dfcombo.loc[row,'range'], wvl)
        
    ## Return headings back to originals and then write to file
    dfcombo.sort_values(['date', 'time'], inplace = True) 
    
    headings29 = ['Temperature(°F)', 'Humidity(%)', 'Wind Speed(m/s)',
                  'Wind Direction From North', 'Barometric Pressure(mm of Hg)', 
                  'Solar Loading(W/m²)',	'Cn2[m^(-2/3)', 'r0']
    combHeadings = ['Date', 'Time', 'Time(s)','range', 'zoom', 'focus', 'image filename',
                    'Image Height', 'Image Width', 'Pixel Step (m)', 'Start (m)', 'Stop (m)', 'obj_size']
    finalcols = combHeadings + headings29
    dfcombo.columns = finalcols
    dfcombo.to_excel(pathout, sheet_name = 'combined_sharpest_images', index = False)



