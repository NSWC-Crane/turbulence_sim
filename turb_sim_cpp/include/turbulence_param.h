#ifndef _TURBULENCE_PARAMETERS_H_
#define _TURBULENCE_PARAMETERS_H_

#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "integrals_spatial_corr.h"

class turbulence_param
{
public:
    std::vector<std::complex<double>> S_vec;
    cv::Mat kernel;
    uint64_t patch_num;

    turbulence_param(uint32_t N_, double D_, double L_, double r0_, double w_, double obj_size_) : N(N_), D(D_), L(L_), r0(r0_), wavelength(w_), obj_size(obj_size_)
    {
        init_params();
    }
    
    //-----------------------------------------------------------------------------
    void init_params(void)
    {
        uint32_t idx;
        double tmp;

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

        generate_psd();

        create_gaussian_kernel(15, 2.8);

        smax_curve.clear();
        for (idx = 1; idx < 101; ++idx)
        {
            tmp = (s_max / (double)(idx)) - 2.0;
            smax_curve.push_back(tmp * tmp);
        }

        // find the argmin of the smax_curve vector
        patch_num = (uint64_t)std::distance(smax_curve.begin(), std::min_element(smax_curve.begin(), smax_curve.end())) + 1;

    }   // end pf init_params
    
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
    inline double get_D(void) { return D; }
    void set_D(double D_) 
    { 
        D = D_; 
        init_params();
    }

    //-----------------------------------------------------------------------------
    inline double get_L(void) { return L; }
    void set_L(double L_) 
    { 
        L = L_; 
        init_params();
    }
   
    //-----------------------------------------------------------------------------
    inline uint64_t get_N(void) { return N; }
    void set_N(uint64_t N_) 
    { 
        N = N_; 
        init_params();
    }

    //-----------------------------------------------------------------------------
    inline double get_wavelength(void) { return wavelength; }
    void set_wavelength(double w_) 
    { 
        wavelength = w_; 
        init_params();
    }

    //-----------------------------------------------------------------------------
    //cv::Mat get_S(void) { return S; }
    //void set_S(cv::Mat S_) { S = S_.clone(); }

    //-----------------------------------------------------------------------------
    inline std::vector<std::complex<double>> get_S_vec(void) { return S_vec; }
    void set_S_vec(std::vector<std::complex<double>> S_vec_) { S_vec = S_vec_; }

    //-----------------------------------------------------------------------------
    inline double get_D_r0(void) { return D_r0; }

    //-----------------------------------------------------------------------------
    inline double get_delta0(void) { return delta0; }

    //-----------------------------------------------------------------------------
    inline double get_scaling(void) { return scaling; }

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

    std::vector<double> smax_curve;
    
    //cv::Mat S;

//-----------------------------------------------------------------------------
//This function generates the PSD necessary for the tilt values (both x and y pixel shifts). The PSD is **4 times**
//the size of the image, this is to simplify the generation of the random vector using a property of Toeplitz
//matrices. This is further highlighted in the genTiltImg() function, where only 1/4 of the entire grid is used
//(this is because of symmetry about the origin -- hence why the PSD is quadruple the size).
//All that is required is the parameter list, p.
//
//adapted from here: 
//https://github.itap.purdue.edu/StanleyChanGroup/TurbulenceSim_v1/blob/master/Turbulence_Sim_v1_python/TurbSim_v1_main.py
//
    void generate_psd(void)
    {
        uint64_t idx;
        cv::Mat x, y;
        cv::Mat s_half;

        // N = 2 * p_obj['N']
        uint32_t N2 = 2 * N;

        double delta0_D = (delta0 / D);
        double s_max_local = delta0_D * (double)N2;
        double i0_val = I0(0);

        // x^(y) = std::exp(y * std::log(x))
        // c1 = 2 * ((24 / 5) * gamma(6 / 5)) * *(5 / 6)
        double c1 = 2.0 * std::exp((5.0 / 6.0) * std::log((24.0 / 5.0) * tgamma(6.0 / 5.0)));

        // c2 = 4 * c1 / np.pi * (gamma(11 / 6)) ** 2
        double c2 = ((4.0 * c1) / CV_PI) * (tgamma(11.0 / 6.0)) * (tgamma(11.0 / 6.0));

        // c3 = (p_obj['Dr0']) ** (5 / 3) / (2 ** (5 / 3)) * (2 * p_obj['wvl'] / (np.pi * p_obj['D'])) ** 2 * 2 * np.pi
        double c3 = 2.0 * CV_PI * std::exp((5.0 / 3.0) * std::log(D_r0 / 2.0)) * (2 * wavelength / (CV_PI * D)) * (2 * wavelength / (CV_PI * D));

        cv::Mat s_arr = linspace(0.0, s_max_local, N2);

        cv::Mat I0_arr = cv::Mat::zeros(s_arr.size(), CV_64FC1);
        cv::Mat I2_arr = cv::Mat::zeros(s_arr.size(), CV_64FC1);

        cv::MatIterator_<double> it, end;
        cv::MatIterator_<double> I0_itr = I0_arr.begin<double>();
        cv::MatIterator_<double> I2_itr = I2_arr.begin<double>();
        for (it = s_arr.begin<double>(), end = s_arr.end<double>(); it != end; ++it, ++I0_itr, ++I2_itr)
        {
            *I0_itr = I0(*it);      // I0_arr[idx] = I0(s_arr[idx])
            *I2_itr = I2(*it);      // I2_arr[idx] = I2(s_arr[idx])
        }

        //[x, y] = np.meshgrid(np.arange(1, N + 0.01, 1), np.arange(1, N + 0.01, 1))
        meshgrid(1.0, (double)N2, N2, 1.0, (double)N2, N2, x, y);

        // i, j = np.int32(N / 2), np.int32(N / 2)
        // s = np.sqrt((x - i) ** 2 + (y - j) ** 2)
        cv::Mat s;
        cv::Mat tmp_x = (x - N).mul(x - N);
        cv::Mat tmp_y = (y - N).mul(y - N);
        cv::sqrt(tmp_x + tmp_y, s);

        // In_1 = In_m(s, p_obj['delta0'] / p_obj['D'] * N, I0_arr)
        cv::Mat In_1 = In_m(s, delta0_D * N2, I0_arr);
        cv::Mat In_2 = In_m(s, delta0_D * N2, I2_arr);
        cv::Mat C = (In_1 + In_2) * (1.0 / i0_val);

        // C[p.get_N(), p.get_N()] = 1
        C.at<double>(N, N) = 1.0;

        //C = C * I0(0) * c2 * (p.get_D_r0()) ** (5.0 / 3.0) / (2 ** (5.0 / 3.0)) * (2 * p.wavelength / (CV_PI * p.D)) ** 2 * 2 * CV_PI;
        C = C * (i0_val * c2 * c3);

        // test of complex vector under the hood
        std::vector<std::complex<double>> c_fft_vec(C.rows * C.cols, 0.0);

        cv::Mat c_fft = cv::Mat(C.rows, C.cols, CV_64FC2, c_fft_vec.data());
        cv::dft(C, c_fft, cv::DFT_COMPLEX_OUTPUT, C.rows);

        S_vec.clear();
        S_vec.resize(C.rows * C.cols);
        s_half = cv::Mat(C.rows, C.cols, CV_64FC2, S_vec.data());
        sqrt_cmplx(c_fft, s_half);

        // find the maximum magnitude of the FFT
        double s_half_max;
        cv::Mat abs_s_half = abs_cmplx(s_half);

        cv::minMaxIdx(abs_s_half, NULL, &s_half_max, NULL, NULL);

        // threshold - all elements < 0.0001 * S_half_max = 0
        threshold_cmplx(abs_s_half, s_half, 0.0001 * s_half_max);

    }   // end of generate_psd

    //-----------------------------------------------------------------------------
    void create_gaussian_kernel(int32_t size, double sigma)
    {
        // assumes a 0 mean Gaussian distribution
        int32_t row, col;
        double s = sigma * sigma;

        kernel = cv::Mat::zeros(size, size, CV_64FC1);

        double t = (1.0 / (2 * CV_PI * s));

        for (row = 0; row < size; ++row)
        {
            for (col = 0; col < size; ++col)
            {
                kernel.at<double>(row, col) = t * std::exp((-((col - (size >> 1)) * (col - (size >> 1))) - ((row - (size >> 1)) * (row - (size >> 1)))) / (2 * s));
            }
        }

        double matsum = (double)cv::sum(kernel)[0];

        kernel = kernel * (1.0 / matsum);	// get the matrix to sum up to 1...

    }	// end of create_gaussian_kernel

};  // end of turbulence_param


#endif  // _TURBULENCE_PARAMETERS_H_
