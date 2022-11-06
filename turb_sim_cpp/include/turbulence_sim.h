#ifndef _TEST_H_TURB_SIM_
#define _TEST_H_TURB_SIM_

#include <cstdint>
#include <cmath>
#include <vector>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "opencv_helper.h"

#include "turbulence_param.h"
#include "integrals_spatial_corr.h"
#include "motion_compensate.h"
#include "zernike_functions.h"

////-----------------------------------------------------------------------------
////This function generates the PSD necessary for the tilt values (both x and y pixel shifts). The PSD is **4 times**
////the size of the image, this is to simplify the generation of the random vector using a property of Toeplitz
////matrices. This is further highlighted in the genTiltImg() function, where only 1/4 of the entire grid is used
////(this is because of symmetry about the origin -- hence why the PSD is quadruple the size).
////All that is required is the parameter list, p.
////
////adapted from here: 
////https://github.itap.purdue.edu/StanleyChanGroup/TurbulenceSim_v1/blob/master/Turbulence_Sim_v1_python/TurbSim_v1_main.py
////
//void generate_psd(turbulence_param &p)
//{
//    uint64_t idx;
//    cv::Mat x, y;
//    cv::Mat s_half;
//
//    uint32_t N = 2 * p.get_N();
//    
//    double delta0_D = (p.get_delta0() / p.get_D());
//    
//    double s_max = delta0_D * (double)N;
//    
//    double i0_val = I0(0);
//
//    // x^(y) = std::exp(y * std::log(x))
//    //double c1 = 2.0 * ((24.0 / 5.0) * tgamma(6.0 / 5.0)) ** (5.0 / 6.0)
//    double c1 = 2.0 * std::exp( (5.0 / 6.0) * std::log((24.0 / 5.0) * tgamma(6.0 / 5.0)));
//    
//    //double c2 = 4.0 * (c1 / CV_PI) * (tgamma(11.0 / 6.0)) ** 2.0
//    double c2 = ((4.0 * c1) / CV_PI) * (tgamma(11.0 / 6.0)) * (tgamma(11.0 / 6.0));
//
//    double c3 = 2.0 * CV_PI * std::exp((5.0 / 3.0) * std::log(p.get_D_r0()/2.0)) * (2 * p.get_wavelength() / (CV_PI * p.get_D())) * (2 * p.get_wavelength() / (CV_PI * p.get_D()));
//    
//    cv::Mat s_arr = linspace(0.0, s_max, N);
//    
//    cv::Mat I0_arr = cv::Mat::zeros(s_arr.size(),CV_64FC1);
//    cv::Mat I2_arr = cv::Mat::zeros(s_arr.size(),CV_64FC1);
//    
//    cv::MatIterator_<double> it, end;
//    cv::MatIterator_<double> I0_itr = I0_arr.begin<double>();
//    cv::MatIterator_<double> I2_itr = I2_arr.begin<double>();
//    for (it = s_arr.begin<double>(), end = s_arr.end<double>(); it != end; ++it, ++I0_itr, ++I2_itr)
//    {
//        //I0_arr[idx] = I0(s_arr[idx])
//        //I2_arr[idx] = I2(s_arr[idx])
//        *I0_itr = I0(*it);
//        *I2_itr = I2(*it);
//    }
//    
//    //i, j = np.int32(N / 2), np.int32(N / 2)
//    
//    //[x, y] = np.meshgrid(np.arange(1, N + 0.01, 1), np.arange(1, N + 0.01, 1))
//    meshgrid(1.0, (double)N, N, 1.0, (double)N, N, x, y);
//    
//    cv::Mat tmp_x = (x - p.get_N()).mul(x - p.get_N());
//    cv::Mat tmp_y = (y - p.get_N()).mul(y - p.get_N());
//
////    cv::Mat s = cv::sqrt((x - p.get_N()) * (x - p.get_N()) + (y - p.get_N()) * (y - p.get_N()));
//    cv::Mat s;
//    cv::sqrt(tmp_x + tmp_y, s);
//     
//    //C = (In_m(s, delta0_D * N , I0_arr) + In_m(s, delta0_D * N, I2_arr)) / I0(0)
//    cv::Mat In_1 = In_m(s, delta0_D * N, I0_arr);
//    cv::Mat In_2 = In_m(s, delta0_D * N, I2_arr);
//    cv::Mat C = (In_1 + In_2) * (1.0 / i0_val);
//
//    // C[p.get_N(), p.get_N()] = 1
//    C.at<double>(p.get_N(), p.get_N()) = 1.0;
//   
//    //C = C * I0(0) * c2 * (p.get_D_r0()) ** (5.0 / 3.0) / (2 ** (5.0 / 3.0)) * (2 * p.wavelength / (CV_PI * p.D)) ** 2 * 2 * CV_PI;
//    C = C * (i0_val * c2 * c3);
//
//    // test of complex vector under the hood
//    std::vector<std::complex<double>> c_fft_vec(C.rows * C.cols, 0.0);
//
//
//    cv::Mat c_fft = cv::Mat(C.rows, C.cols, CV_64FC2, c_fft_vec.data());
//    cv::dft(C, c_fft, cv::DFT_COMPLEX_OUTPUT, C.rows);
//    
//    p.S_vec.resize(C.rows * C.cols);
//    s_half = cv::Mat(C.rows, C.cols, CV_64FC2, p.S_vec.data());
//    sqrt_cmplx(c_fft, s_half);
//
//    // find the maximum magnitude of the FFT
//    double s_half_max;
//    cv::Mat abs_s_half = abs_cmplx(s_half);
//
//    cv::minMaxIdx(abs_s_half, NULL, &s_half_max, NULL, NULL);
//
//    // threshold - all elements < 0.0001 * S_half_max = 0
//    threshold_cmplx(abs_s_half, s_half, 0.0001 * s_half_max);
// /*   
//    S_half[np.abs(S_half) < 0.0001 * S_half_max] = 0
//*/
//
//    p.set_S(s_half);
//
//}   // end of generate_psd


//-----------------------------------------------------------------------------
void centroid_psf(cv::Mat &psf, double threshold = 0.95)
{
    int32_t radius = 2;
    int32_t cx, cy;
    int32_t x, y, w, h;
    double temp_sum = 0.0;
    cv::Mat mx, my;
    cv::Mat tmp_psf;
    cv::Rect psf_roi;
    std::vector<std::vector<cv::Point> > psf_contours;
    std::vector<cv::Vec4i> psf_hr;

    double psf_sum = cv::sum(psf)[0];
    psf *= 1.0 / psf_sum;

    //x = np.linspace(0, psf.shape[0], psf.shape[0])
    //y = np.linspace(0, psf.shape[1], psf.shape[1])
    //col, row = np.meshgrid(x, y)
    meshgrid<double>(0, psf.cols - 1, psf.cols, 0, psf.rows - 1, psf.rows, mx, my);

    try 
    {
        //cen_row = np.uint8(np.sum(row * psf))
        //cen_col = np.uint8(np.sum(col * psf))
        cx = (uint8_t)cv::sum(mx.mul(psf))[0];
        cy = (uint8_t)cv::sum(my.mul(psf))[0];
        //cv::Mat psf_t;
        //cv::threshold(psf, psf_t, 0.001, 255, cv::THRESH_BINARY);
        //psf_t.convertTo(psf_t, CV_8UC1);
        //cv::findContours(psf_t, psf_contours, psf_hr, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);





        while (temp_sum < threshold)
        {
            ++radius;

            x = std::max(0, cx - radius);
            y = std::max(0, cy - radius);
            w = (cx + radius + 1) < psf.cols ? (2*radius + 1) : w + 1;
            h = (cy + radius + 1) < psf.rows ? (2*radius + 1) : h + 1;

            //    return_psf = psf[cen_row - radius:cen_row + radius + 1, cen_col - radius : cen_col + radius + 1]
            psf_roi = cv::Rect(x, y, w, h);
            tmp_psf = psf(psf_roi);
            //    temp_sum = np.sum(return_psf)
            temp_sum = cv::sum(tmp_psf)[0];
            //    #print(radius, temp_sum)
        }

        psf = tmp_psf.clone();
    }
    catch (std::exception e)
    {
        std::cout << "error: " << e.what() << std::endl;
    }
}   // end of centroid_psf

//-----------------------------------------------------------------------------
//This function takes the p_obj(with the PSD!) and applies it to the image.If no PSD is found, one will be
//generated.However, it is** significantly** faster to generate the PSD once and then use it to draw the values from.
//This is also done automatically, because it is significantly faster.
//
// param img: The input image(assumed to be N x N pixels)
// param p_obj: The parameter object -- with the PSD is preferred
// return: The output, tilted image
//
// adapted from here:
// https://github.itap.purdue.edu/StanleyChanGroup/TurbulenceSim_v1/blob/master/Turbulence_Sim_v1_python/TurbSim_v1_main.py
//
void generate_tilt_image(cv::Mat& src, turbulence_param &p, cv::RNG& rng, cv::Mat& dst)
{
    uint64_t idx;
    //double c1 = 2 * std::sqrt(2) * p.get_N() * (p.get_L() / p.get_delta0());
    double c1 = 2.0*std::sqrt(2) * p.get_N() * (p.get_L() / p.get_delta0());
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
    cv::filter2D(mv_xc, mv_xc, CV_64FC1, p.kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT_101);
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
    cv::filter2D(mv_yc, mv_yc, CV_64FC1, p.kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT_101);
    mv_yc -= cv::mean(mv_yc)[0];
    //#MVy = 1 / p_obj['scaling'] * MVy[0:p_obj['N'], round(p_obj['N'] / 2) : 2 * p_obj['N'] - round(p_obj['N'] / 2)]
    
    //img_ = motion_compensate(img, MVx - np.mean(MVx), MVy - np.mean(MVy), 0.5)
    motion_compensate(src, dst, mv_xc, mv_yc, 0.5);

    //#plt.quiver(MVx[::10, ::10], MVy[::10, ::10], scale = 60)
    //#plt.show()

}   // end of generate_tilt_image

//-----------------------------------------------------------------------------
//adapted from here :
//https://github.itap.purdue.edu/StanleyChanGroup/TurbulenceSim_v1/blob/master/Turbulence_Sim_v1_python/TurbSim_v1_main.py
void generate_blur_image(cv::Mat& src, turbulence_param &p, cv::RNG& rng, cv::Mat& dst)
{
    uint64_t idx;
    //uint64_t patch_num = 0;
    uint64_t N = p.get_N();
    double tmp_min = std::numeric_limits<double>::max();
    double tmp;
    double psf_sum;
    double z_i = 1.2;
    double pad_size = 0.0;

    cv::Mat patch_mask;
    cv::Mat psf, temp_psf;
    cv::Mat img_patches = cv::Mat::zeros(N, N, CV_64FC1);
    cv::Mat tmp_conv;

    std::vector<double> coeff;

    uint32_t NN = 28;    // default=32, For extreme scenarios, this may need to be increased

    //    smax = p_obj['delta0'] / p_obj['D'] * p_obj['N']
    //double smax = (p.get_delta0() / p.get_D()) * p.get_N();

    //    temp = np.arange(1, 101)
    //cv::Mat tmp = linspace(1.0, 100.0, 100);

    //    patchN = temp[np.argmin((smax * np.ones(100) / temp - 2) * *2)]
    //for (idx = 1; idx < 101; ++idx)
    //{
    //    tmp = (smax / (double)(idx)) - 1.5;
    //    tmp *= tmp;
    //    if (tmp < tmp_min)
    //    {
    //        patch_num = idx;
    //        tmp_min = tmp;
    //    }
    //}
       
    //    patch_size = round(p_obj['N'] / patchN)
    double patch_size = std::floor((N / p.patch_num) + 0.5);
     
    //    xtemp = np.round_(p_obj['N'] / (2 * patchN) + np.linspace(0, p_obj['N'] - p_obj['N'] / patchN + 0.001, patchN)  )
    cv::Mat x_tmp = linspace(0.0, (double)(N - N / (double)p.patch_num), p.patch_num) + N/(double)(2* p.patch_num);
    x_tmp = round(x_tmp);

    //    xx, yy = np.meshgrid(xtemp, xtemp)
    cv::Mat xx, yy;
    meshgrid(x_tmp, x_tmp, xx, yy);
    
    //    xx_flat, yy_flat = xx.flatten(), yy.flatten()
    xx = xx.reshape(1, xx.total());
    yy = yy.reshape(1, yy.total());
    
    //    img_patches = np.zeros((p_obj['N'], p_obj['N'], int(patchN * *2)))
    //    den = np.zeros((p_obj['N'], p_obj['N']))
    dst = cv::Mat::zeros(N, N, CV_64FC1);
    cv::Mat den = cv::Mat(N, N, CV_64FC1, cv::Scalar::all(1.0e-6));
        
    //    patch_indx, patch_indy = np.meshgrid(np.linspace(-patch_size, patch_size + 0.001, num = 2 * patch_size + 1), np.linspace(-patch_size, patch_size + 0.001, num = 2 * patch_size + 1))
    cv::Mat patch_indx, patch_indy;
    meshgrid(-patch_size, patch_size, 2 * patch_size + 1, -patch_size, patch_size, 2 * patch_size + 1, patch_indx, patch_indy);
    patch_indx = patch_indx.mul(patch_indx);
    patch_indx *= (-1.0 / (patch_size * patch_size));
    patch_indy = patch_indy.mul(patch_indy);
    patch_indy *= (-1.0 / (patch_size * patch_size));
    cv::exp(patch_indx, patch_indx);
    cv::exp(patch_indy, patch_indy);
    // np.ones((patch_size * 2 + 1, patch_size * 2 + 1))

    cv::Mat exp_tmp = patch_indx.mul(patch_indy);
    //exp_tmp = exp_tmp.mul(cv::Mat::ones(patch_size * 2 + 1, patch_size * 2 + 1, CV_64FC1));
    //cv::flip(exp_tmp, exp_tmp, -1);

    cv::Mat k2 = (1 / 9.0) * cv::Mat::ones(3, 3, CV_64FC1);

    //    for i in range(int(patchN * *2)) :
    for (idx = 0; idx < (p.patch_num * p.patch_num); ++idx)
    {
        // aa = genZernikeCoeff(36, p_obj['Dr0'])
        generate_zernike_coeff(36, p.get_D_r0(), coeff, rng);

        // temp, x, y, nothing, nothing2 = psfGen(NN, coeff = aa, L = p_obj['L'], D = p_obj['D'], z_i = 1.2, wavelength = p_obj['wvl'])
        generate_psf(NN, p, coeff, temp_psf, z_i, pad_size);
        
        // psf = np.abs(temp) * *2
        psf = abs_cmplx(temp_psf);
        psf = psf.mul(psf);
        
        // psf = psf / np.sum(psf.ravel())
        psf_sum = cv::sum(psf)[0];
        psf *= 1.0 / psf_sum;
        //centroid_psf(psf, 0.95);
        
        // # focus_psf, _, _ = centroidPsf(psf, 0.95) : Depending on the size of your PSFs, you may want to use this
        // psf = resize(psf, (round(NN / p_obj['scaling']), round(NN / p_obj['scaling'])))
        cv::resize(psf, psf, cv::Size(std::floor(NN / p.get_scaling() + 0.5), std::floor(NN / p.get_scaling() + 0.5)), 0.0, 0.0, cv::INTER_LINEAR);
        //cv::flip(psf, psf, -1);
        //cv::filter2D(psf, psf, -1, k2, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT(0));
        //cv::GaussianBlur(psf, psf, cv::Size(3, 3), 0);

        // patch_mask = np.zeros((p_obj['N'], p_obj['N']))
        patch_mask = cv::Mat::zeros(N, N, CV_64FC1);

        // patch_mask[round(xx_flat[i]), round(yy_flat[i])] = 1
//        patch_mask.at<double>((uint64_t)(*yy.ptr<double>(idx) + 0.5), (uint64_t)(*xx.ptr<double>(idx) + 0.5)) = 1.0;
        patch_mask.at<double>((uint64_t)(*xx.ptr<double>(idx) + 0.5), (uint64_t)(*yy.ptr<double>(idx) + 0.5)) = 1.0;

        // patch_mask = scipy.signal.fftconvolve(patch_mask, np.exp(-patch_indx * *2 / patch_size * *2) * np.exp(-patch_indy * *2 / patch_size * *2) * np.ones((patch_size * 2 + 1, patch_size * 2 + 1)), mode = 'same')
        cv::filter2D(patch_mask, patch_mask, -1, exp_tmp, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT(0));

        // den += scipy.signal.fftconvolve(patch_mask, psf, mode = 'same')
        cv::filter2D(patch_mask, tmp_conv, -1, psf, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT_101);
        //cv::Mat tmp_conv2;
        //cv::filter2D(patch_mask, tmp_conv2, -1, psf, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT(0));
        den += tmp_conv;

        // img_patches[:, : , i] = scipy.signal.fftconvolve(img * patch_mask, psf, mode = 'same')
        patch_mask = patch_mask.mul(src);
        cv::filter2D(patch_mask, tmp_conv, -1, psf, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT_101);
        img_patches += tmp_conv;

    }
    // out_img = np.sum(img_patches, axis = 2) / (den + 0.000001)
    dst = img_patches.mul(1.0 / den);

}   // end of generate_blur_image



#endif  // _TEST_H_TURB_SIM_

