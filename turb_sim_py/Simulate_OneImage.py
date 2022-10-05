# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 16:08:05 2022

@author: victoria.lockridge
"""

'''
Demo of Multi-Aperture Turbulence Simulation.

Nicholas Chimitt and Stanley H. Chan "Simulating anisoplanatic turbulence
by sampling intermodal and spatially correlated Zernike coefficients," Optical Engineering 59(8), Aug. 2020

ArXiv:  https://arxiv.org/abs/2004.11210

Nicholas Chimitt and Stanley Chan
Copyright 2021
Purdue University, West Lafayette, In, USA.

MODIFIED:  V Lockridge, NSWC Crane
Simulate one image from modified baseline image.
'''


from matplotlib import pyplot as plt
import numpy as np
import TurbSim_v1_main as util
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os


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
        print("Error:  max and min are ", (img.max(),img.min()) )
        

D = 0.095
wvl = 0.525e-6 
fileA = r"C:\Data\JSSAP\combined_sharpest_images_withAtmos.xlsx"

if __name__ == '__main__':
    dfcombo = pd.read_excel(fileA)
    colnames = ['date','time','timeSecs', 'range', 'zoom', 'focus', 'imageFn', 'imageHt', 
            'imageWd', 'pixelStep', 'start','stop','obj_size', 'temp', 'humidity', 
            'windSpeed', 'windDir', 'barPressure', 'solarLoad', 'cn2','r0'] 
    dfcombo.columns = colnames

    modFile = r"C:\Data\JSSAP\modifiedBaselines\Mod_baseline_z3500_r0900.png"
    img = cv2.imread(modFile)
    img = img[:,:,1]  # Select green channel
    rng = 900
    zoomV = 3500
    dfview = dfcombo[(dfcombo.loc[:,'range'] == rng) & (dfcombo.loc[:,'zoom'] == zoomV)] 
    dfview.reset_index(inplace = True)
    cn2 = dfview.loc[0,'cn2']
    obj_size = dfview.loc[0,'obj_size']
    N = dfview.loc[0,'imageHt']    
    r0 = CalculateR0(cn2, rng, wvl)
     
    param_obj = util.p_obj(N, D, rng, r0, wvl, obj_size)
    S = util.gen_PSD(param_obj)     # finding the PSD, see the def for details
    param_obj['S'] = S
    img_tilt, _ = util.genTiltImg(img, param_obj) 
    img_blur = util.genBlurImage(param_obj, img_tilt)
    
    realFile = r"C:\Data\JSSAP\sharpest\z3500\0900\image_z03497_f47470_e02743_i11.png"
    realImg = cv2.imread(realFile)
    realImg = realImg[:,:,1]  # Select green channel
    
    plt.subplot(2, 2, 1)
    plt.imshow(img_tilt)
    
    plt.subplot(2, 2, 2)
    plt.imshow(img_blur)
    
    plt.subplot(2, 2, 3)
    plt.imshow(img)
    
    plt.subplot(2, 2, 4)
    plt.imshow(realImg)

# Use cv2 to display images    
#    montage = np.concatenate((img_tilt, img_blur), axis=1)
#    cv2.imshow('Image', montage)
#   # cv2.resizeWindow('Image', 600, 300)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
#    cv2.namedWindow('tilt window', cv2.WINDOW_KEEPRATIO)
#    cv2.imshow('tilt window', img_blur)
#    cv2.resizeWindow('tilt window', 600, 600)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
#    # To save images, use ChangImgDtypeUint8 for images with pixel values
#    #          between 0 and 1.
#    #  Note:  define output filenames and directories
#    img_blur256 = ChangImgDtypeUint8(img_blur)
#    cv2.imwrite(os.path.join(dirOut, imSoutC),img_blur256) 

    

    

