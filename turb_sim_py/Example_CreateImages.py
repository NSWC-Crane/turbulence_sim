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
from CreateInfoTable import CreateInfoDf
import os
import cv2


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
dirn = r"C:\Data\JSSAP\sharpest"
dirOut = r"C:\Data\JSSAP\Purdue\0914"

if __name__ == '__main__':
    dfcomb = CreateInfoDf()
    for row in range(len(dfcomb)):
        if dfcomb.loc[row,'range'] < 1000:
            dfcomb.loc[row,'baselineImg'] = 'baseline_z' +  str(dfcomb.loc[row,'zoom']) + '_r0'\
                                  + str(dfcomb.loc[row,'range']) + '.png'
        else:
            dfcomb.loc[row,'baselineImg'] = 'baseline_z' +  str(dfcomb.loc[row,'zoom']) + '_r'\
                                  + str(dfcomb.loc[row,'range']) + '.png'
    for row in range(len(dfcomb)):  #range(51,53,1):#range(len(dfcomb)):    
        N = dfcomb.loc[row,'imageHt']
        L = dfcomb.loc[row,'range']
        obj_size = dfcomb.loc[row,'obj_size']
        cn2 = dfcomb.loc[row,'cn2']
        dirnB = os.path.join(dirn, 'z' + str(dfcomb.loc[row,'zoom']))
        baseImg = os.path.join(dirnB, dfcomb.loc[row,'baselineImg'])
        print('ZOOM ', str(dfcomb.loc[row,'zoom']))
        print('RANGE ', str(L))
        
        if L < 1000:
            dirnR = os.path.join(dirnB, '0' + str(L))
        else:
            dirnR = os.path.join(dirnB, str(L))
        imgFn = dfcomb.loc[row,'imageFn']
        imgFn = imgFn.split('/')[1]
        realImg = os.path.join(dirnR, imgFn)
        #realImg = realImg.replace('/','\\')
        # Get green channel only
        img = plt.imread(baseImg)[:,:,1]
        realI = plt.imread(realImg)[:,:,1]
        
        r0 = CalculateR0(cn2, L, wvl)
        param_obj = util.p_obj(N, D, L, r0, wvl, obj_size)
        print(param_obj)
        
        S = util.gen_PSD(param_obj)     # finding the PSD, see the def for details
        param_obj['S'] = S  
        
        img_tilt, _ = util.genTiltImg(img, param_obj)       # generating the tilt-only image
        img_blur = util.genBlurImage(param_obj, img_tilt)
        
        ##############
        
        r0 = CalculateR0(cn2, L, wvl)
       
        param_obj = util.p_obj(N, D, L, r0, wvl, obj_size)
        print(param_obj)
        
        S = util.gen_PSD(param_obj)     # finding the PSD, see the def for details
        param_obj['S'] = S  
        
        img_tilt, _ = util.genTiltImg(img, param_obj)       # generating the tilt-only image
        img_blur = util.genBlurImage(param_obj, img_tilt)
        
        # Save all images
        cn2str = str(cn2)
        cn2str = cn2str.replace('.','p')
        cn2str = cn2str.replace('-','')
        figout = 'FIG_z' + str(dfcomb.loc[row,'zoom']) + 'r' + str(L) + 'cn2_' + cn2str + '_Green.png'
        imBout = 'z' + str(dfcomb.loc[row,'zoom']) + 'r' + str(L) + 'cn2_' + cn2str + '_Green_Img.png'
        imSout = 'z' + str(dfcomb.loc[row,'zoom']) + 'r' + str(L) + 'cn2_' + cn2str + '_Green_SimImg.png'
        imRout = 'z' + str(dfcomb.loc[row,'zoom']) + 'r' + str(L) + 'cn2_' + cn2str + '_Green_Real.png'  

        imBoutC = 'z' + str(dfcomb.loc[row,'zoom']) + 'r' + str(L) + 'cn2_' + cn2str + '_GreenC_Img.png'
        imSoutC = 'z' + str(dfcomb.loc[row,'zoom']) + 'r' + str(L) + 'cn2_' + cn2str + '_GreenC_SimImg.png'
        imRoutC = 'z' + str(dfcomb.loc[row,'zoom']) + 'r' + str(L) + 'cn2_' + cn2str + '_GreenC_Real.png'               
        
#        imageio.imwrite(os.path.join(dirOut, imBout),np.uint8(img))
#        imageio.imwrite('result.png',np.uint8(img))
#        imageio.imwrite('result.png',np.uint8(img))
        
#        plt.imsave(os.path.join(dirOut, imBout),img, cmap = 'gray')
#        plt.imsave(os.path.join(dirOut, imSout),img_blur, cmap = 'gray')
#        plt.imsave(os.path.join(dirOut, imRout),realI, cmap = 'gray')

        
        img256 = ChangImgDtypeUint8(img)
        cv2.imwrite(os.path.join(dirOut, imBoutC),img256)  
        img_blur256 = ChangImgDtypeUint8(img_blur)
        cv2.imwrite(os.path.join(dirOut, imSoutC),img_blur256) 
        real256 = ChangImgDtypeUint8(realI)
        cv2.imwrite(os.path.join(dirOut, imRoutC),real256)  
               
        fig = plt.figure(figsize=(15, 5))
        
        fig.add_subplot(1, 4,  1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        #plt.imshow(img,  vmin=0, vmax=1)
        plt.title('img')
    
        fig.add_subplot(1, 4, 2)
        plt.imshow(img_blur, cmap='gray', vmin=0, vmax=1)
        #plt.imshow(img_blur, vmin=0, vmax=1)
        plt.title('img_tilt & img_blur')
        
        fig.add_subplot(1, 4, 3)
        plt.imshow(realI, cmap='gray', vmin=0, vmax=1)
        #plt.imshow(realI, vmin=0, vmax=1)
        plt.title('real img')
        
        plt.show()
        
        pathOut = os.path.join(dirOut, figout)
        fig.savefig(pathOut)
        
