#ifndef _TEST_H_TURB_SIM_
#define _TEST_H_TURB_SIM_

#include <cstdint>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "opencv_helper.h"

#include "integrals_spatial_corr.h"


class param_obj
{
public:

    param_obj(uint32_t N_, double D_, double L_, double r0_, double w_, double obj_size_) : N(N_), D(D_), L(L_), r0(r0_), wavelength(w_), obj_size(obj_size_)
    {
        init_params();
    }
    
    //-----------------------------------------------------------------------------
    void init_params(void)
    {
        D_r0 = D / r0;
        delta0 = (L * wavelength) / (2.0 * D);
        k = (2.0 * CV_PI) / wavelength;
        s_max = (delta0 / D) * (double)N;   //???????
        spacing = delta0 / D;
        ob_s = obj_size;
        scaling = obj_size / (N * delta0);

        s_max *= scaling;
        spacing *= scaling;
        ob_s *= scaling;
        delta0 *= scaling;
    }
    
    //-----------------------------------------------------------------------------
    void set_N(uint64_t N_)
    {
        N = N_;
        init_params();
    }
    
    //-----------------------------------------------------------------------------
    void set_D(double D_)
    {
        D = D_;
        init_params();
    }    

    //-----------------------------------------------------------------------------
    void set_L(double L_)
    {
        L = L_;
        init_params();
    }
    
    //-----------------------------------------------------------------------------
    void update_params(uint32_t N_, double D_, double L_, double r0_, double w_, double obj_size_)
    {
        N = N_;
        D = D_;
        L = L_;
        r0 = r0_;
        wavelength = w_;
        obj_size = obj_size_;

        init_params();
        
    }

    //-----------------------------------------------------------------------------
    void update_params(double L_)
    {
        L = L_;

        init_params();
    }

    //-----------------------------------------------------------------------------
    void update_params(double L_, double D_)
    {
        D = D_;
        L = L_;

        init_params();
        
    }
    
    //-----------------------------------------------------------------------------
    double get_delta0(void) { return delta0; }

    //-----------------------------------------------------------------------------
    double get_D(void) { return D; }

    //-----------------------------------------------------------------------------
    double get_D_r0(void) { return D_r0; }
    
    //-----------------------------------------------------------------------------
    uint64_t get_N(void) { return N; }

    //-----------------------------------------------------------------------------
    double get_wavelength(void) { return wavelength; }

//-----------------------------------------------------------------------------
private:
    uint64_t N;             // size of pixels of one image dimension (assumed to be square image N x N)
    double D;               // size of aperture diameter (meters)
    double L;               // length of propagation (meters)
    double r0;              // Fried parameter (meters)
    double wavelength;      // wavelength (meters)
    double obj_size;
    
    double D_r0;
    double delta0;
    double k;
    double s_max;
    double spacing;
    double ob_s;
    double scaling;
    
};


//-----------------------------------------------------------------------------
/*
This function generates the PSD necessary for the tilt values (both x and y pixel shifts). The PSD is **4 times**
the size of the image, this is to simplify the generation of the random vector using a property of Toeplitz
matrices. This is further highlighted in the genTiltImg() function, where only 1/4 of the entire grid is used
(this is because of symmetry about the origin -- hence why the PSD is quadruple the size).
All that is required is the parameter list, p.

adapted from here: 
https://github.itap.purdue.edu/StanleyChanGroup/TurbulenceSim_v1/blob/master/Turbulence_Sim_v1_python/TurbSim_v1_main.py

*/

void generate_psd(param_obj p, cv::Mat &s_half)
{
    uint64_t idx;
    cv::Mat x, y;
    
    uint32_t N = 2 * p.get_N();
    
    double delta0_D = (p.get_delta0() / p.get_D());
    
    double s_max = delta0_D * (double)N;
    
    double i0_val = I0(0);

    // x^(y) = std::exp(y * std::log(x))
    //double c1 = 2.0 * ((24.0 / 5.0) * tgamma(6.0 / 5.0)) ** (5.0 / 6.0)
    double c1 = 2.0 * std::exp( (5.0 / 6.0) * std::log((24.0 / 5.0) * tgamma(6.0 / 5.0)));
    
    //double c2 = 4.0 * (c1 / CV_PI) * (tgamma(11.0 / 6.0)) ** 2.0
    double c2 = ((4.0 * c1) / CV_PI) * (tgamma(11.0 / 6.0)) * (tgamma(11.0 / 6.0));

    double c3 = 2.0 * CV_PI * std::exp((5.0 / 3.0) * std::log(p.get_D_r0()/2.0)) * (2 * p.get_wavelength() / (CV_PI * p.get_D())) * (2 * p.get_wavelength() / (CV_PI * p.get_D()));
    
    cv::Mat s_arr = linspace(0.0, s_max, N);
    
    cv::Mat I0_arr = cv::Mat::zeros(s_arr.size(),CV_64FC1);
    cv::Mat I2_arr = cv::Mat::zeros(s_arr.size(),CV_64FC1);
    
    cv::MatIterator_<double> it, end;
    cv::MatIterator_<double> I0_it = I0_arr.begin<double>();
    cv::MatIterator_<double> I2_it = I2_arr.begin<double>();
    for (it = s_arr.begin<double>(), end = s_arr.end<double>(); it != end; ++it)
    {
        //I0_arr[idx] = I0(s_arr[idx])
        //I2_arr[idx] = I2(s_arr[idx])
        *I0_it = I0(*it);
        *I2_it = I2(*it);

        ++I0_it;
        ++I2_it;
    }
    
    //i, j = np.int32(N / 2), np.int32(N / 2)
    
    //[x, y] = np.meshgrid(np.arange(1, N + 0.01, 1), np.arange(1, N + 0.01, 1))
    meshgrid(1.0, (double)N, N, 1.0, (double)N, N, x, y);
    

    cv::Mat tmp_x = (x - p.get_N()).mul(x - p.get_N());
    cv::Mat tmp_y = (y - p.get_N()).mul(y - p.get_N());

//    cv::Mat s = cv::sqrt((x - p.get_N()) * (x - p.get_N()) + (y - p.get_N()) * (y - p.get_N()));
    cv::Mat s;
    cv::sqrt(tmp_x + tmp_y, s);
     
    //C = (In_m(s, delta0_D * N , I0_arr) + In_m(s, delta0_D * N, I2_arr)) / I0(0)
    cv::Mat In_1 = In_m(s, delta0_D * N, I0_arr);
    cv::Mat In_2 = In_m(s, delta0_D * N, I2_arr);
    cv::Mat C = (In_1 + In_2) * (1.0 / i0_val);

    // C[p.get_N(), p.get_N()] = 1
    C.at<double>(p.get_N(), p.get_N()) = 1.0;
   
    //C = C * I0(0) * c2 * (p.get_D_r0()) ** (5.0 / 3.0) / (2 ** (5.0 / 3.0)) * (2 * p.wavelength / (CV_PI * p.D)) ** 2 * 2 * CV_PI;
    C = C * (i0_val * c2 * c3);

    cv::Mat c_fft = cv::Mat::zeros(C.rows, C.cols, CV_64FC1);
    cv::dft(C, c_fft, cv::DFT_COMPLEX_OUTPUT, C.rows);

    //cv::Mat c_fft = np.fft.fft2(C)
    
    s_half = sqrt_cmplx(c_fft);

    // find the maximum magnitude of the FFT
    double s_half_max;
    cv::Mat abs_s_half = abs_cmplx(s_half);

    cv::minMaxIdx(abs_s_half, NULL, &s_half_max, NULL, NULL);

    threshold_cmplx(abs_s_half, s_half, 0.0001 * s_half_max);
 /*   
    // threshold - all elements < 0.0001 * S_half_max = 0
    S_half[np.abs(S_half) < 0.0001 * S_half_max] = 0
*/

    //return S_half
}





#endif  // _TEST_H_TURB_SIM_

