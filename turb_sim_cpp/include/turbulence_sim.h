#ifndef _TEST_H_TURB_SIM_
#define _TEST_H_TURB_SIM_

#include <cstdint>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "opencv_helper.h"

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
    void update_params(double L_, double r0_)
    {
        D = D_;
        L = L_;

        init_params();
        
    }
    
    //-----------------------------------------------------------------------------
    double get_delta0(void) { return delta0; }

    //-----------------------------------------------------------------------------
    double get_D_r0(void) { return D_r0; }
    
    //-----------------------------------------------------------------------------
    uint64_t get_N(void) { return N; }

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------


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

adapted from here: https://github.itap.purdue.edu/StanleyChanGroup/TurbulenceSim_v1/blob/master/Turbulence_Sim_v1_python/TurbSim_v1_main.py

*/

//def gen_PSD(p):
void generate_psd(param_obj p, cv::Mat &s_half)
{
    uint64_t idx;
    cv::Mat x, y;
    
    uint64_t N = 2 * p.get_N();
    
    double delta0_D = (p.get_delta0() / p.D);
    
    double s_max = delta0_D * (double)N;
    

    // x^(y) = std::exp(y * std::log(x))
    //double c1 = 2.0 * ((24.0 / 5.0) * tgamma(6.0 / 5.0)) ** (5.0 / 6.0)
    double c1 = 2.0 * std::exp( (5.0 / 6.0) * std::log((24.0 / 5.0) * tgamma(6.0 / 5.0)));
    
    //double c2 = 4.0 * (c1 / CV_PI) * (tgamma(11.0 / 6.0)) ** 2.0
    double c2 = ((4.0 * c1) / CV_PI) * (tgamma(11.0 / 6.0)) * (tgamma(11.0 / 6.0));
    
    cv::Mat s_arr = linspace(0, s_max, s_max/(double)N);
    
    cv::Mat I0_arr = cv::Mat::zeros(s_arr.size(),CV_64FC1);
    cv::Mat I2_arr = cv::Mat::zeros(s_arr.size(),CV_64FC1);
        
    for(idx=0; idx<s_arr.length(); ++idx)
    {
        //I0_arr[idx] = I0(s_arr[idx])
        //I2_arr[idx] = I2(s_arr[idx])
    }
    
    //i, j = np.int32(N / 2), np.int32(N / 2)
    
    //[x, y] = np.meshgrid(np.arange(1, N + 0.01, 1), np.arange(1, N + 0.01, 1))
    meshgrid(1, N, 1, 1, N, 1, x, y);
    

    cv::Mat s = cv::sqrt((x - p.get_N()) * (x - p.get_N()) + (y - p.get_N()) * (y - p.get_N()));
    
    /*    
    C = (In_m(s, delta0_D * N , I0_arr) + In_m(s, delta0_D * N, I2_arr)) / I0(0)
    
    C[p.get_N(), p.get_N()] = 1
    
    C = C * I0(0) * c2 * (p.get_D_r0()) ** (5.0 / 3.0) / (2 ** (5.0 / 3.0)) * (2 * p.wavelength / (CV_PI * p.D)) ** 2 * 2 * CV_PI;
    
    cv::Mat c_fft = np.fft.fft2(C)
    
    cv::Mat s_half = cv::sqrt(Cfft);
    
    // find the maximum magnitude of the FFT
    double s_half_max = np.max(np.max(np.abs(s_half)))
    
    // threshold - all elements < 0.0001 * S_half_max = 0
    S_half[np.abs(S_half) < 0.0001 * S_half_max] = 0
*/

    //return S_half
}





#endif  // _TEST_H_TURB_SIM_

