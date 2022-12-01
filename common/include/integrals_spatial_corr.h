#ifndef _TS_INTEGRAL_SPATIAL_CORRELATION_H_
#define _TS_INTEGRAL_SPATIAL_CORRELATION_H_

#include <cstdint>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "opencv_helper.h"

#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_integration.h>

//-----------------------------------------------------------------------------
struct Ix_params {
	double s;
	double c1;
};


// integration function: f(z) = z**(-14/3)) * jv(0,2*s*z) * (jv(2,z)**2)
double f_I0(double z, void* p)
{
	Ix_params& params = *reinterpret_cast<Ix_params*>(p);
	double j0 = gsl_sf_bessel_J0(params.s * z);
	double j2 = gsl_sf_bessel_Jn(2, z);
	return j0 * j2 * j2 * std::exp(params.c1 * std::log(z));
}

// integration function: f(z) = z**(-14/3)) * jv(2,2*s*z) * (jv(2,z)**2)
double f_I2(double z, void* p)
{
	Ix_params& params = *reinterpret_cast<Ix_params*>(p);
	double j0 = gsl_sf_bessel_Jn(2, params.s * z);
	double j2 = gsl_sf_bessel_Jn(2, z);
	return j0 * j2 * j2 * std::exp(params.c1 * std::log(z));
}

double I0(double s)
{
	uint32_t idx;
	uint32_t n = 100;
	double z = 0.0;
	double I0_s = 0.0;

	double j0 = 0.0;
	double j2 = 0.0;
	double t;

	double c1 = -14.0 / 3.0;
	// the integration step size
	double delta_z = (1.0 / (double)n);
	//s = 2.0 * s;

	// integration function: f(z) = z**(-14/3)) * jv(0,2*s*z) * (jv(2,z)**2
	// integration: f(t/(1-t))/((1-t)^2)dt, 0 -> 1 ---> t = 1/(z+1) --> t/(1-t) --> 1/z
	// x^(y) = std::exp(y * std::log(x))


	Ix_params params;
	params.s = 2.0 * s;
	params.c1 = -14.0 / 3.0;

	gsl_function F;
	F.function = &f_I0;
	F.params = reinterpret_cast<void*>(&params);

	double result, error;
	size_t neval = 1000;
	gsl_integration_workspace* w = gsl_integration_workspace_alloc(neval);

	//const double xlow = 0;
	//const double xhigh = 1e6;
	const double epsabs = 1e-5;
	const double epsrel = 1e-5;

	//int gsl_integration_qagiu(gsl_function *f, double a, double epsabs, double epsrel, size_t limit, gsl_integration_workspace *workspace, double *result, double *abserr)
	int code = gsl_integration_qagiu(&F, 0.0, epsabs, epsrel, neval, w, &I0_s, &error);

	gsl_integration_workspace_free(w);

/*
	// start the simpson rule integration: f(x0)
	I0_s = 0.0;
	z += delta_z;

	for (idx=1; idx<n; idx += 2)
	{
		t = 1.0 / (z + 1);
		j0 = gsl_sf_bessel_J0(s * (1.0/z));
		j2 = gsl_sf_bessel_Jn(2, (1.0/z));
		I0_s += 4.0 * j0 * j2 * j2 * std::exp(c1 * std::log(1.0/z)) * (1.0/(1.0-t))*(1.0/(1.0-t));
		z += delta_z;

		t = 1.0 / (z + 1);
		j0 = gsl_sf_bessel_J0(s * (1.0/z));
		j2 = gsl_sf_bessel_Jn(2, (1.0 / z));
		I0_s += 2.0 * j0 * j2 * j2 * std::exp(c1 * std::log(1.0 / z)) * (1.0 / (1.0 - t)) * (1.0 / (1.0 - t));
		z += delta_z;
	}

	t = 1.0 / (z + 1);
	j0 = gsl_sf_bessel_J0(s * (1.0 / z));
	j2 = gsl_sf_bessel_Jn(2, (1.0 / z));
	I0_s += j0 * j2 * j2 * std::exp(c1 * std::log(1.0 / z)) * (1.0 / (1.0 - t)) * (1.0 / (1.0 - t));
	I0_s *= 3.0 / delta_z;
*/

	return I0_s;
}


//-----------------------------------------------------------------------------
double I2(double s)
{
	uint32_t idx;
	uint32_t n = 10000;
	double z = 0.0;
	double I2_s = 0.0;

	double j0 = 0.0;
	double j2 = 0.0;
	double t;

	double c1 = -14.0 / 3.0;
	// the integration step size
	double delta_z = (1.0 - 0.0) / (double)n;

	// integration function: f(z) = z**(-14/3)) * jv(2,2*s*z) * (jv(2,z)**2
	// integration: f(t/(1-t))/((1-t)^2)dt, 0 -> 1 ---> t = 1/(z+1)
	// x^(y) = std::exp(y * std::log(x))


	Ix_params params;
	params.s = 2.0 * s;
	params.c1 = -14.0 / 3.0;

	gsl_function F;
	F.function = &f_I2;
	F.params = reinterpret_cast<void*>(&params);

	double result, error;
	size_t neval = 1000;
	gsl_integration_workspace* w = gsl_integration_workspace_alloc(neval);

	//const double xlow = 0;
	//const double xhigh = 1e6;
	const double epsabs = 1e-5;
	const double epsrel = 1e-5;

	//int gsl_integration_qagiu(gsl_function *f, double a, double epsabs, double epsrel, size_t limit, gsl_integration_workspace *workspace, double *result, double *abserr)
	int code = gsl_integration_qagiu(&F, 0.0, epsabs, epsrel, neval, w, &I2_s, &error);

	gsl_integration_workspace_free(w);

	return I2_s;
}

//-----------------------------------------------------------------------------
/*
def In_m(s, spacing, In_arr):
	idx = np.int32(np.floor(s.flatten()/spacing))
	M,N = np.shape(s)[0], np.shape(s)[1]
	In = np.reshape(np.take(In_arr, idx), [M,N])

	return In
*/

cv::Mat In_m(cv::Mat &s, double spacing, cv::Mat& src)
{
	// idx = np.int32(np.floor(s.flatten() / spacing))
	//uint32_t idx = floor(s/spacing)
	
	// M, N = np.shape(s)[0], np.shape(s)[1]
	uint32_t M = s.rows;
	uint32_t N = s.cols;
	cv::Mat dst = cv::Mat::zeros(M, N, CV_64FC1);

	cv::Mat tmp;
	s.convertTo(tmp, CV_64FC1, 1.0 / spacing);

	// In = np.reshape(np.take(In_arr, idx), [M, N])
	cv::MatIterator_<double> tmp_itr = tmp.begin<double>();
	cv::MatIterator_<double> tmp_end = tmp.end<double>();
	cv::MatIterator_<double> dst_itr = dst.begin<double>();

	double* src_ptr = src.ptr<double>(0);

	for ( ; tmp_itr != tmp_end; ++tmp_itr, ++dst_itr)
	{
		*dst_itr = src_ptr[(uint64_t)(*tmp_itr)];
	}

	return dst;
}

#endif  // _TS_INTEGRAL_SPATIAL_CORRELATION_H_
