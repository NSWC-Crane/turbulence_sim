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
#include "motion_compensate.h"

class param_obj
{
public:
    std::vector<std::complex<double>> S_vec;

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
    double get_D(void) { return D; }
    void set_D(double D_) 
    { 
        D = D_; 
        init_params();
    }

    //-----------------------------------------------------------------------------
    double get_L(void) { return L; }
    void set_L(double L_) 
    { 
        L = L_; 
        init_params();
    }
   
    //-----------------------------------------------------------------------------
    uint64_t get_N(void) { return N; }
    void set_N(uint64_t N_) 
    { 
        N = N_; 
        init_params();
    }

    //-----------------------------------------------------------------------------
    double get_wavelength(void) { return wavelength; }
    void set_wavelength(double w_) 
    { 
        wavelength = w_; 
        init_params();
    }

    //-----------------------------------------------------------------------------
    cv::Mat get_S(void) { return S; }
    void set_S(cv::Mat S_) { S = S_.clone(); }

    //-----------------------------------------------------------------------------
    std::vector<std::complex<double>> get_S_vec(void) { return S_vec; }
    void set_S_vec(std::vector<std::complex<double>> S_vec_) { S_vec = S_vec_; }

    //-----------------------------------------------------------------------------
    double get_D_r0(void) { return D_r0; }

    //-----------------------------------------------------------------------------
    double get_delta0(void) { return delta0; }

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
    
    cv::Mat S;


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

void generate_psd(param_obj &p)
{
    uint64_t idx;
    cv::Mat x, y;
    cv::Mat s_half;

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

    // test of complex vector under the hood
    std::vector<std::complex<double>> c_fft_vec(C.rows * C.cols, 0.0);


    cv::Mat c_fft = cv::Mat(C.rows, C.cols, CV_64FC2, c_fft_vec.data());
    cv::dft(C, c_fft, cv::DFT_COMPLEX_OUTPUT, C.rows);
    

    p.S_vec.resize(C.rows * C.cols);
    s_half = cv::Mat(C.rows, C.cols, CV_64FC2, p.S_vec.data());
    sqrt_cmplx(c_fft, s_half);

    // find the maximum magnitude of the FFT
    double s_half_max;
    cv::Mat abs_s_half = abs_cmplx(s_half);

    cv::minMaxIdx(abs_s_half, NULL, &s_half_max, NULL, NULL);

    // threshold - all elements < 0.0001 * S_half_max = 0
    threshold_cmplx(abs_s_half, s_half, 0.0001 * s_half_max);
 /*   
    S_half[np.abs(S_half) < 0.0001 * S_half_max] = 0
*/

    p.set_S(s_half);
}

//-----------------------------------------------------------------------------
/*
This function takes the p_obj(with the PSD!) and applies it to the image.If no PSD is found, one will be
generated.However, it is** significantly** faster to generate the PSD onceand then use it to draw the values from.
This is also done automatically, because it is significantly faster.

: param img : The input image(assumed to be N x N pixels)
: param p_obj : The parameter object -- with the PSD is preferred
: return : The output, tilted image

adapted from here :
https://github.itap.purdue.edu/StanleyChanGroup/TurbulenceSim_v1/blob/master/Turbulence_Sim_v1_python/TurbSim_v1_main.py
*/

void generate_tilt_image(cv::Mat& src, param_obj p, cv::RNG& rng, cv::Mat& dst)
{
    uint64_t idx;
    double c1 = std::sqrt(2) * 2 * p.get_N() * (p.get_L() / p.get_delta0());
    uint64_t N = 2 * p.get_N();
    uint64_t N_2 = p.get_N() >> 1;
    uint64_t N2 = N * N;

    std::complex<double> tmp;

    cv::Mat mv_x;
    cv::Mat mv_y;
    cv::Mat rnd_x(N, N, CV_64FC1);
    cv::Mat rnd_y(N, N, CV_64FC1);
    cv::Mat S = cv::Mat::zeros(N, N, CV_64FC2);
    cv::MatIterator_<double> rnd_itr;
    cv::MatIterator_<cv::Vec2d> S_itr;

    //MVx = np.real(np.fft.ifft2(p_obj['S'] * np.random.randn(2 * p_obj['N'], 2 * p_obj['N']))) * np.sqrt(2) * 2 * p_obj['N'] * (p_obj['L'] / p_obj['delta0'])
    rng.fill(rnd_x, cv::RNG::NORMAL, 0.0, 1.0);
    rnd_itr = rnd_x.begin<double>();
    S_itr = S.begin<cv::Vec2d>();
    for (idx = 0; idx < (N * N); ++idx, ++rnd_itr, ++S_itr)
    {
        tmp = p.S_vec[idx] * (*rnd_itr);
        *S_itr = cv::Vec2d(tmp.real(), tmp.imag());
    }

    cv::dft(S, mv_x, cv::DFT_INVERSE + cv::DFT_SCALE, S.rows);


    //MVx = MVx[round(p_obj['N'] / 2):2 * p_obj['N'] - round(p_obj['N'] / 2), 0 : p_obj['N']]
    cv::Mat mv_xc = mv_x(cv::Rect(N_2, 0, p.get_N(), p.get_N()));
    mv_xc = c1 * get_real(mv_xc);
    mv_xc -= cv::mean(mv_xc)[0];

    //#MVx = 1 / p_obj['scaling'] * MVx[round(p_obj['N'] / 2):2 * p_obj['N'] - round(p_obj['N'] / 2), 0 : p_obj['N']]
    //MVy = np.real(np.fft.ifft2(p_obj['S'] * np.random.randn(2 * p_obj['N'], 2 * p_obj['N']))) * np.sqrt(2) * 2 * p_obj['N'] * (p_obj['L'] / p_obj['delta0'])
    rng.fill(rnd_y, cv::RNG::NORMAL, 0.0, 1.0);
    
    rnd_itr = rnd_y.begin<double>();
    S_itr = S.begin<cv::Vec2d>();

    for (idx = 0; idx < (N * N); ++idx, ++rnd_itr, ++S_itr)
    {
        tmp = p.S_vec[idx] * (*rnd_itr);
        *S_itr = cv::Vec2d(tmp.real(), tmp.imag());
    }

    cv::dft(S, mv_y, cv::DFT_INVERSE + cv::DFT_SCALE, S.rows);

    //MVy = MVy[0:p_obj['N'], round(p_obj['N'] / 2) : 2 * p_obj['N'] - round(p_obj['N'] / 2)]
    cv::Mat mv_yc = mv_y(cv::Rect(0, N_2, p.get_N(), p.get_N()));
    mv_yc = c1 * get_real(mv_yc);
    mv_yc -= cv::mean(mv_yc)[0];
    //#MVy = 1 / p_obj['scaling'] * MVy[0:p_obj['N'], round(p_obj['N'] / 2) : 2 * p_obj['N'] - round(p_obj['N'] / 2)]
    
    //img_ = motion_compensate(img, MVx - np.mean(MVx), MVy - np.mean(MVy), 0.5)
    motion_compensate(src, dst, mv_xc, mv_yc, 0.5);

    //#plt.quiver(MVx[::10, ::10], MVy[::10, ::10], scale = 60)
    //#plt.show()

}   // end of generate_tilt_image


//def genBlurImage(p_obj, img) :
void generate_blur_image(cv::Mat& src, param_obj p, cv::RNG& rng, cv::Mat& dst)
{
    //    smax = p_obj['delta0'] / p_obj['D'] * p_obj['N']
    double smax = (p.get_delta0() / p.get_D()) * p.get_N();

    //    temp = np.arange(1, 101)
    cv::Mat tmp = linspace(1.0, 100.0, 100);

    //    patchN = temp[np.argmin((smax * np.ones(100) / temp - 2) * *2)]
    
    
    //    patch_size = round(p_obj['N'] / patchN)
    //    xtemp = np.round_(p_obj['N'] / (2 * patchN) + np.linspace(0, p_obj['N'] - p_obj['N'] / patchN + 0.001, patchN))
    //    xx, yy = np.meshgrid(xtemp, xtemp)
    //    xx_flat, yy_flat = xx.flatten(), yy.flatten()
    //    NN = 32 # For extreme scenarios, this may need to be increased
    //    img_patches = np.zeros((p_obj['N'], p_obj['N'], int(patchN * *2)))
    //    den = np.zeros((p_obj['N'], p_obj['N']))
    //    patch_indx, patch_indy = np.meshgrid(np.linspace(-patch_size, patch_size + 0.001, num = 2 * patch_size + 1), np.linspace(-patch_size, patch_size + 0.001, num = 2 * patch_size + 1))
    //
    //    for i in range(int(patchN * *2)) :
    //        aa = genZernikeCoeff(36, p_obj['Dr0'])
    //        temp, x, y, nothing, nothing2 = psfGen(NN, coeff = aa, L = p_obj['L'], D = p_obj['D'], z_i = 1.2, wavelength = p_obj['wvl'])
    //        psf = np.abs(temp) * *2
    //        psf = psf / np.sum(psf.ravel())
    //        # focus_psf, _, _ = centroidPsf(psf, 0.95) : Depending on the size of your PSFs, you may want to use this
    //        psf = resize(psf, (round(NN / p_obj['scaling']), round(NN / p_obj['scaling'])))
    //        patch_mask = np.zeros((p_obj['N'], p_obj['N']))
    //        patch_mask[round(xx_flat[i]), round(yy_flat[i])] = 1
    //        patch_mask = scipy.signal.fftconvolve(patch_mask, np.exp(-patch_indx * *2 / patch_size * *2) * np.exp(-patch_indy * *2 / patch_size * *2) * np.ones((patch_size * 2 + 1, patch_size * 2 + 1)), mode = 'same')
    //        den += scipy.signal.fftconvolve(patch_mask, psf, mode = 'same')
    //        img_patches[:, : , i] = scipy.signal.fftconvolve(img * patch_mask, psf, mode = 'same')
    //
    //        out_img = np.sum(img_patches, axis = 2) / (den + 0.000001)
    //        return out_img
}



#endif  // _TEST_H_TURB_SIM_

