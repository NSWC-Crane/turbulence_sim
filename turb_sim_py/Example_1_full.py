'''
Demo of Multi-Aperture Turbulence Simulation.

Nicholas Chimitt and Stanley H. Chan "Simulating anisoplanatic turbulence
by sampling intermodal and spatially correlated Zernike coefficients," Optical Engineering 59(8), Aug. 2020

ArXiv:  https://arxiv.org/abs/2004.11210

Nicholas Chimitt and Stanley Chan
Copyright 2021
Purdue University, West Lafayette, In, USA.
'''

from matplotlib import pyplot as plt
import numpy as np
import TurbSim_v1_main as util


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = rgb2gray(plt.imread('../data/checker_board_32x32.png'))

N = 16             # size of the image -- assumed to be square (pixels)
D = 0.095           # length of aperture diameter (meters)
L = 1000            # length of propagation (meters)

wvl = 0.525e-6      # the mean wavelength -- typically somewhere suitably in the middle of the spectrum will be sufficient
r0 = 0.0386         # the Fried parameter r0. The value of D/r0 is critically important! (See associated paper)
                    # All values for wvl = 0.525e-6: cn = 1e-15 -> r0 = 0.1535, Cn = 1e-14 -> r0 = 0.0386, Cn = 1e-13 -> r0 = 0.0097

pixel = 0.0125
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
fig = plt.figure(figsize=(columns, rows))

for i in range(1):
    img_tilt, _ = util.genTiltImg(img, param_obj)       # generating the tilt-only image
    img_blur = util.genBlurImage(param_obj, img_tilt)

    fig.add_subplot(1, 2,  1)
    plt.imshow(img_tilt, cmap='gray', vmin=0, vmax=1)
    plt.title('img_tilt')

    fig.add_subplot(1, 2, 2)
    plt.imshow(img_blur, cmap='gray', vmin=0, vmax=1)
    plt.title('img_tilt & img_blur')
    plt.show()

# breakpoint before code exists
bp = 1
