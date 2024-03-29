'''
Demo of Multi-Aperture Turbulence Simulation.

Nicholas Chimitt and Stanley H. Chan "Simulating anisoplanatic turbulence
by sampling intermodal and spatially correlated Zernike coefficients," Optical Engineering 59(8), Aug. 2020

ArXiv:  https://arxiv.org/abs/2004.11210

Nicholas Chimitt and Stanley Chan
Copyright 2021
Purdue University, West Lafayette, In, USA.
'''
import cv2
from matplotlib import pyplot as plt
import numpy as np
import TurbSim_v1_main as util
import time

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# img = rgb2gray(plt.imread('../data/checker_board_32x32.png'))
filename = 'C:/Projacts/data/turbulence/sharpest/z5000/baseline_z5000_r1000.png'
# filename = "C:/Projects/data/turbulence/sharpest/z5000/baseline_z5000_r1000.png"

img = (plt.imread(filename))[:,:,1]

N = img.shape[0]             # size of the image -- assumed to be square (pixels)
D = 0.095           # length of aperture diameter (meters)
L = 1000            # length of propagation (meters)

wvl = 0.525e-6      # the mean wavelength -- typically somewhere suitably in the middle of the spectrum will be sufficient

Cn2 = 1e-13
k = 2 * np.pi / wvl

# the Fried parameter r0. The value of D/r0 is critically important! (See associated paper)
# All values for wvl = 0.525e-6: cn = 1e-15 -> r0 = 0.1535, Cn = 1e-14 -> r0 = 0.0386, Cn = 1e-13 -> r0 = 0.0097
r0 = np.exp(-0.6 * np.log(0.158625 * k * k * Cn2 * L))

pixel = 0.004217
obj_size = N * pixel   # the size of the object in the object plane (meters). Can be different the Nyquist sampling, scaling
                    # will be done automatically.

param_obj = util.p_obj(N, D, L, r0, wvl, obj_size) # generating the parameter object, some other things are computed within this
                                                   # function, see the def for details

#print(param_obj['N'] * param_obj['delta0'], param_obj['smax'], param_obj['scaling'])
print(param_obj)

#print('hi')
#print(param_obj['spacing'])
#print('hi')

S = util.gen_PSD(param_obj)     # finding the PSD, see the def for details
param_obj['S'] = S              # appending the PSD to the parameter object for convenience

# setting values to rows and column variables
rows = 5
columns = 10

# create figure
# fig = plt.figure(figsize=(columns, rows))

# img = img[16:16+N, 16:16+N]

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

# for i in range(1):
while(1):
    tic = time.perf_counter()
    img_tilt, _ = util.genTiltImg(img, param_obj)       # generating the tilt-only image

    # fig.add_subplot(1, 2,  1)
    # plt.imshow(img_tilt, cmap='gray', vmin=0, vmax=1)
    # plt.title('img_tilt')

    img_blur = util.genBlurImage(param_obj, img_tilt)
    # img_blur = util.genBlurImage(param_obj, img)

    toc = time.perf_counter()
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(img_blur, cmap='gray', vmin=0, vmax=1)
    # plt.title('img_tilt & img_blur')
    # plt.show()

    print(f"time (s):  {toc - tic:0.6f} ")

    montage = np.concatenate((img_tilt, img_blur), axis=1)
    cv2.imshow('Image', montage)
    cv2.waitKey(10)


cv2.destroyAllWindows()

# breakpoint before code exists
bp = 1
