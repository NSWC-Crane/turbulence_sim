import os
import platform

# os.add_dll_directory(r"D:/Projects/vcpkg/installed/x64-windows/bin")
# os.add_dll_directory(r"C:/Projects/turbulence_sim/turb_sim_lib/build/Release")

from cffi import FFI

from matplotlib import pyplot as plt
import numpy as np
import cv2
import time

script_path = os.path.realpath(os.path.dirname(__file__))
ffi = FFI()

def prepend_path_env(added_paths, to_env='PATH'):
    path_sep = ';'
    prior_path_env = os.environ.get(to_env)
    prior_paths = prior_path_env.split(path_sep)
    added_paths = [x for x in added_paths if os.path.exists(x)]
    new_paths = prior_paths + added_paths
    new_env_val = path_sep.join(new_paths)
    return new_env_val

##-----------------------------------------------------------------------------
# modify these to point to the right locations
if platform.system() == "Windows":
    libname = "turb_sim.dll"
    home = script_path[0:2]         # assumes that this project is placed into the same root folder as the library project
    lib_location = home + "/Projects/turbulence_sim/turb_sim_lib/build/Release/" + libname
    # do this for some reason the dlopen hates this dll
    os.chdir(script_path + "/../turb_sim_lib/build/Release")

elif platform.system() == "Linux":
    libname = "libturb_sim.so"
    home = os.path.expanduser('~')
    lib_location = home + "/Projects/turbulence_sim/turb_sim_lib/build/" + libname
else:
    quit()

# open the library and keep as a global variable
print(lib_location)
turb_lib = ffi.dlopen(lib_location)

# this declares the functions that will be used in the library (taken directly from the library header)
ffi.cdef('''

void init_turbulence_params(unsigned int N_, double D_, double L_, double Cn2_, double w_, double obj_size_);
void apply_turbulence(unsigned int img_w, unsigned int img_h, double *img_, double *turb_img_);

''')

def init_turb_lib(N, D, L, Cn2, wvl, obj_size):
    global turb_lib

    turb_lib.init_turbulence_params(int(N), D, L, Cn2, wvl, obj_size)

def apply_turbulence(img):
    global turb_lib

    img_h = img.shape[0]
    img_w = img.shape[1]

    blur_img = np.zeros([img_h, img_w], dtype=np.float64)
    blur_img_t = ffi.cast("double *", blur_img.ctypes.data)
    img_t = ffi.cast("double *", img.ctypes.data)

    turb_lib.apply_turbulence(img_w, img_h, img_t, blur_img_t)

    return blur_img.astype(np.uint8)

if __name__ == '__main__':

    # img = rgb2gray(plt.imread('../data/checker_board_32x32.png'))
    filename = 'C:/Projects/data/turbulence/sharpest/z5000/baseline_z5000_r1000.png'
    # filename = "D:/data/turbulence/sharpest/z5000/baseline_z5000_r1000.png"

    img = 255 * (plt.imread(filename))[:, :, 1]
    img = img.astype(np.float64)

    N = img.shape[0]  # size of the image -- assumed to be square (pixels)
    D = 0.095  # length of aperture diameter (meters)
    L = 1000  # length of propagation (meters)

    wvl = 0.525e-6  # the mean wavelength -- typically somewhere suitably in the middle of the spectrum will be sufficient

    Cn2 = 1e-13

    # the Fried parameter r0. The value of D/r0 is critically important! (See associated paper)
    # All values for wvl = 0.525e-6: cn = 1e-15 -> r0 = 0.1535, Cn = 1e-14 -> r0 = 0.0386, Cn = 1e-13 -> r0 = 0.0097

    pixel = 0.004217
    obj_size = N * pixel  # the size of the object in the object plane (meters). Can be different the Nyquist sampling, scaling
    # will be done automatically.

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    init_turb_lib(N, D, L, Cn2, wvl, obj_size)

    while(1):
        blur_img = apply_turbulence(img)

        cv2.imshow('Image', blur_img)
        cv2.waitKey(10)

    cv2.destroyAllWindows()


