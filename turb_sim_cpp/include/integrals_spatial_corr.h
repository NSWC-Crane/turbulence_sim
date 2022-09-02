#ifndef _TS_INTEGRAL_SPATIAL_CORRELATION_H_
#define _TS_INTEGRAL_SPATIAL_CORRELATION_H_

#include <cstdint>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "opencv_helper.h"


//-----------------------------------------------------------------------------
def I0(s):
	# z = np.linspace(1e-6, 1e3, 1e5)
	# f_z = (z**(-14/3))*jv(0,2*s*z)*(jv(2,z)**2)
	# I0_s = np.trapz(f_z, z)

	I0_s, _ = integrate.quad( lambda z: (z**(-14/3))*jv(0,2*s*z)*(jv(2,z)**2), 1e-4, np.inf, limit = 100000)
	# print('I0: ',I0_s)
	return I0_s

//-----------------------------------------------------------------------------
def I2(s):
	# z = np.linspace(1e-6, 1e3, 1e5)
	# f_z = (z**(-14/3))*jv(2,2*s*z)*(jv(2,z)**2)
	# I2_s = np.trapz(f_z, z)

	I2_s, _ = integrate.quad( lambda z: (z**(-14/3))*jv(2,2*s*z)*(jv(2,z)**2), 1e-4, np.inf, limit = 100000)
	# print('I2: ',I2_s)
	return I2_s

//-----------------------------------------------------------------------------
def In_m(s, spacing, In_arr):
	idx = np.int32(np.floor(s.flatten()/spacing))
	M,N = np.shape(s)[0], np.shape(s)[1]
	In = np.reshape(np.take(In_arr, idx), [M,N])

	return In
    
#endif  // _TS_INTEGRAL_SPATIAL_CORRELATION_H_
