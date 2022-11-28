# -*- coding: utf-8 -*-
"""
Using Purdue code that simulates images with tilt and blur with D Emerson's improvements.
Created on Mon Oct 24 09:59:25 2022

Creates simulated images at specific range/zoom values with varying Cn2/r0 values.
Images are based on Modified Baseline images (no blur).
Images are saved for use in MATLAB to run similarity metrics.

V Lockridge
"""

from matplotlib import pyplot as plt
import numpy as np
import TurbSim_v1_main as util
import os
import cv2
import pandas as pd

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
cn2_1 = [5e-16,1e-15,4e-15,7e-15,1e-14,2e-14,3e-14,4e-14,5e-14,6e-14,7e-14,8e-14,9e-14,
       1e-13,4e-13,7e-13,1e-12]
# Add another set
cn2_2 = [2e-13,3e-13,5e-13,6e-13,8e-13,9e-13,2e-12,3e-12,4e-12,5e-12,6e-12,7e-12,8e-12,9e-12,1e-11]
cn2_3 = [2e-15,3e-15,5e-15,6e-15,8e-15,9e-15,6e-16,7e-16,8e-16,9e-16,4e-16]
numSims = 20  # Number of simulated images of same zoom/range/cn2 - will average metrics in MATLAB.

if __name__ == '__main__':
    # Define directories
    platform = str(os.getenv("PLATFORM"))
    if (platform == "Laptop"):
        data_root = r"D:\data\turbulence"
    elif platform == "LaptopN":
        data_root = r"C:\Projects\data\turbulence"
    else:   
        data_root = r"C:\Data\JSSAP"
    
    fileA = os.path.join(data_root, 'combined_sharpest_images_withAtmos.xlsx')
    dirnB = os.path.join(data_root,'modifiedBaselines')  # Location of modified baseline images
    dirOut = os.path.join(data_root, r"modifiedBaselines\SimImgs_VaryingCn2") # Save images here

    # Select each modified baseline image in dir C:\Data\JSSAP\modifiedBaselines
    # Create several simulated images (numSims) using r0 for appropriatge range and cn2 values
    # Save images to use in MATLAB to calculate similarity metrics
    
    # Import file with obj_size information - required to create simulated images
    dfAtm = pd.read_excel(fileA)
    colnames = ['date','time','timeSecs', 'range', 'zoom', 'focus', 'imageFn', 'imageHt', 
            'imageWd', 'pixelStep', 'start','stop','obj_size', 'temp', 'humidity', 
           'windSpeed', 'windDir', 'barPressure', 'solarLoad', 'cn2','r0'] 
    dfAtm.columns = colnames
    
    # Select each .png in dirnB directory, create simulated image using its range
    #          the cn2 values listed in the above list called cn2 
    for file in os.listdir(dirnB):
        if file.endswith(".png"):
            fileN = os.path.join(dirnB, file)
            print(os.path.join(dirnB, file))
            # Pull range and zoom values from file name
            rng = int((fileN.split("r",1)[1]).split(".",1)[0])
            zoom = int((fileN.split("z",1)[1]).split("_",1)[0])
            print("range ", rng, " zoom ", zoom)
            
            imgB = plt.imread(fileN)
            [M,N] = imgB.shape
            obj_size =  dfAtm[(dfAtm.range == rng) & (dfAtm.zoom == zoom)].reset_index().obj_size[0]
 
            r0s = [CalculateR0(cn2_3[i],rng,wvl) for i in range(len(cn2_3))]
            
            # Create simulated image for each cn2/r0 listed
            for i,r0 in enumerate(r0s):
            
                param_obj = util.p_obj(N, D, rng, r0, wvl, obj_size)
                S = util.gen_PSD(param_obj)     # finding the PSD, see the def for details
                param_obj['S'] = S  
                for k in range(numSims):
                    img_tilt, _ = util.genTiltImg(imgB, param_obj)    # generating the tilt-only image
                    img_blurtilt = util.genBlurImage(param_obj, img_tilt)  
                
                    # Save simulated image in directory dirOut
                    cn2str = str(cn2_3[i]).replace('.','p').replace('-','')
                    imSimOut = "r" + str(rng) + "_z" + str(zoom) + "_c" + cn2str + "_N" + str(k) + ".png"

                    # Change images to 256 to use cv2 to imwrite
                    img_blurtilt256 = ChangImgDtypeUint8(img_blurtilt)
                    cv2.imwrite(os.path.join(dirOut, imSimOut),img_blurtilt256) 
    
    # Save information on range, cn2, r0 for MATLAB work
    # Set up dataframe cols:  rng cn2 r0
#    cn2 = [5e-16,1e-15,4e-15,7e-15,1e-14,2e-14,3e-14,4e-14,5e-14,6e-14,7e-14,8e-14,9e-14,
#       1e-13,4e-13,7e-13,1e-12, 2e-13,3e-13,5e-13,6e-13,8e-13,9e-13,2e-12,3e-12,4e-12,5e-12,
#       6e-12,7e-12,8e-12,9e-12,1e-11]
#    
    cn2 = cn2_1 + cn2_2 + cn2_3

    dfwrite = pd.DataFrame()
    rangeV = [600+50*i for i in range(9)] 
    numc = len(cn2)
    rcls = [numc * [rangeV[i]] for i in range(len(rangeV))]
    dfwrite['range'] = [j for i in rcls for j in i]
    dfwrite['cn2'] = len(rangeV)*cn2
    dfwrite['r0'] = CalculateR0(dfwrite.cn2,dfwrite.range, wvl)
    # Save range, cn2, r0 values in .csv file for each range for use in MATLAB 
    #                program that calculates metrics.
    dfwrite.to_csv(os.path.join(dirOut, 'turbNums.csv'),index = False)


