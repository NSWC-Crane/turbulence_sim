'''
Demo of Multi-Aperture Turbulence Simulation.

Nicholas Chimitt and Stanley H. Chan "Simulating anisoplanatic turbulence
by sampling intermodal and spatially correlated Zernike coefficients," Optical Engineering 59(8), Aug. 2020

ArXiv:  https://arxiv.org/abs/2004.11210

Nicholas Chimitt and Stanley Chan
Copyright 2021
Purdue University, West Lafayette, In, USA.

MODIFIED:  V Lockridge, NSWC Crane
'''

"""
TO DO
1.  Use opencv to import, write, filter 

"""

from matplotlib import pyplot as plt
import numpy as np
import TurbSim_v1_main as util
import os
import cv2
import pandas as pd


# VICKY ADDED
def CalculateR0(cn2, L, wvl):
    k = 2*np.pi/wvl
    b0 = 0.158625
    r_inside = b0*k**2*cn2*L
    r0 = np.power(r_inside,-0.6)   # or -3/5
    return r0

def ChangImgDtypeUint8(img):
    
    if img.max() <= 1.0 and img.min() >=0.0:
        imgInt = img*255
        imgInt = imgInt.astype(np.uint8)
        return imgInt
        
    else:
        print("Warning:  max and min are ", (img.max(),img.min()) )
        

D = 0.095
wvl = 0.525e-6 
dirnB = r"C:\Data\JSSAP\modifiedBaselines"
dirnRB = r"C:\Data\JSSAP\sharpest"
dirOut = r"C:\Data\JSSAP\modifiedBaselines\SimImages2"
fileA = r"C:\Data\JSSAP\combined_sharpest_images_withAtmos.xlsx"

if __name__ == '__main__':
    # Create dataframe of atmosheric data for each file by zoom/range
    dfcomb = pd.read_excel(fileA)
    colnames = ['date','time','timeSecs', 'range', 'zoom', 'focus', 'imageFn', 'imageHt', 
            'imageWd', 'pixelStep', 'start','stop','obj_size', 'temp', 'humidity', 
            'windSpeed', 'windDir', 'barPressure', 'solarLoad', 'cn2','r0'] 
    dfcomb.columns = colnames
    
    # Add file names of baseline images to dfcomb
    for row in range(len(dfcomb)):
        if dfcomb.loc[row,'range'] < 1000:
            dfcomb.loc[row,'baselineImg'] = 'Mod_baseline_z' +  str(dfcomb.loc[row,'zoom']) + '_r0'\
                                  + str(dfcomb.loc[row,'range']) + '.png'
        else:
            dfcomb.loc[row,'baselineImg'] = 'Mod_baseline_z' +  str(dfcomb.loc[row,'zoom']) + '_r'\
                                  + str(dfcomb.loc[row,'range']) + '.png'
                                  
    for row in range(len(dfcomb)): 
        # Get data to pass to util.gen_PSD
        N = dfcomb.loc[row,'imageHt']
        L = dfcomb.loc[row,'range']
        obj_size = dfcomb.loc[row,'obj_size']
        cn2 = dfcomb.loc[row,'cn2']
        #dirnB = dirn #os.path.join(dirn, 'z' + str(dfcomb.loc[row,'zoom']))
        dirnR = os.path.join(dirnRB, 'z' + str(dfcomb.loc[row,'zoom']))
        pathBaseImg = os.path.join(dirnB, dfcomb.loc[row,'baselineImg'])
        print('ZOOM ', str(dfcomb.loc[row,'zoom']))
        print('RANGE ', str(L))
        
        if L < 1000:
            dirnRF = os.path.join(dirnR, '0' + str(L))
        else:
            dirnRF = os.path.join(dirnR, str(L))
        fileImg = dfcomb.loc[row,'imageFn']
        fileImg = fileImg.split('/')[1]
        fileReal = os.path.join(dirnRF, fileImg)
        # Get green channel only
        img = plt.imread(pathBaseImg) #[:,:,1] Modified base is only green channel
        realI = plt.imread(fileReal)[:,:,1]        
        
        r0 = CalculateR0(cn2, L, wvl)
        param_obj = util.p_obj(N, D, L, r0, wvl, obj_size)
        print(param_obj)
        
        S = util.gen_PSD(param_obj)     # finding the PSD, see the def for details
        param_obj['S'] = S  
        img_tilt, _ = util.genTiltImg(img, param_obj)       # generating the tilt-only image
        img_blurtilt = util.genBlurImage(param_obj, img_tilt)       
        img_blurOnly = util.genBlurImage(param_obj, img)

        # Save all images
        cn2str = str(cn2)
        cn2str = cn2str.replace('.','p')
        cn2str = cn2str.replace('-','')
        figout = 'FIG_z' + str(dfcomb.loc[row,'zoom']) + 'r' + str(L) + 'cn2_' + cn2str + '_Green.png'

        imBoutC = 'Mz' + str(dfcomb.loc[row,'zoom']) + 'r' + str(L) + 'cn2_' + cn2str + '_GreenC_Img.png'
        imSoutC = 'Mz' + str(dfcomb.loc[row,'zoom']) + 'r' + str(L) + 'cn2_' + cn2str + '_GreenC_SimImg.png'
        imRoutC = 'Mz' + str(dfcomb.loc[row,'zoom']) + 'r' + str(L) + 'cn2_' + cn2str + '_GreenC_Real.png'
        imBlurC = 'Mz' + str(dfcomb.loc[row,'zoom']) + 'r' + str(L) + 'cn2_' + cn2str + '_GreenC_BlurOnly.png'                       
        
        # Change images to 256 to use cv2 to imwrite
        img256 = ChangImgDtypeUint8(img)
        img_blurtilt256 = ChangImgDtypeUint8(img_blurtilt)
        img_blurOnly256 = ChangImgDtypeUint8(img_blurOnly)
        imgReal256 = ChangImgDtypeUint8(realI)
        
        cv2.imwrite(os.path.join(dirOut, imBoutC),img256)  
        cv2.imwrite(os.path.join(dirOut, imSoutC),img_blurtilt256) 
        cv2.imwrite(os.path.join(dirOut, imBlurC),img_blurOnly256) 
        cv2.imwrite(os.path.join(dirOut, imRoutC),imgReal256)  
               
        fig = plt.figure(figsize=(20, 5))

        fig.add_subplot(1, 4,  1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title('Modified Baseline')
    
        fig.add_subplot(1, 4, 2)
        plt.imshow(img_blurtilt, cmap='gray', vmin=0, vmax=1)
        plt.title('Simulated:  Blur and Tilt')
        
        fig.add_subplot(1, 4, 3)
        plt.imshow(img_blurOnly, cmap='gray', vmin=0, vmax=1)
        plt.title('Simulated:  Blur Only')
        
        fig.add_subplot(1, 4, 4)
        plt.imshow(realI, cmap='gray', vmin=0, vmax=1)
        plt.title('Real Image')
        
        fig.suptitle('Zoom ' + str(dfcomb.loc[row,'zoom']) + ' Range ' + str(L), fontsize = 16)
        plt.show()
        
        pathOut = os.path.join(dirOut, figout)
        fig.savefig(pathOut)
        
