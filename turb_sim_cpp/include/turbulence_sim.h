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

//-----------------------------------------------------------------------------
void centroid_psf(cv::Mat &psf, double threshold = 0.95)
{
    int32_t radius = 4;
    int32_t cx, cy;
    int32_t x, y;
    int32_t w = (2 * radius + 1), h = (2 * radius + 1);

    double temp_sum = 0.0;
    cv::Mat mx, my;
    cv::Mat tmp_psf;
    cv::Rect psf_roi;
    std::vector<std::vector<cv::Point> > psf_contours;
    std::vector<cv::Vec4i> psf_hr;

    //x = np.linspace(0, psf.shape[0], psf.shape[0])
    //y = np.linspace(0, psf.shape[1], psf.shape[1])
    //col, row = np.meshgrid(x, y)
    meshgrid<double>(0, psf.cols - 1, psf.cols, 0, psf.rows - 1, psf.rows, mx, my);

    try 
    {
        cx = (uint8_t)cv::sum(mx.mul(psf))[0];
        cy = (uint8_t)cv::sum(my.mul(psf))[0];

        while (temp_sum < threshold)
        {
            ++radius;

            x = std::max(0, cx - radius);
            y = std::max(0, cy - radius);
            w = (cx + radius + 1) <= psf.cols ? (2*radius) : psf.cols - x;
            h = (cy + radius + 1) <= psf.rows ? (2*radius) : psf.rows - y;

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
    uint64_t min_x, max_x, min_kx, max_kx;
    uint64_t min_y, max_y, min_ky, max_ky;

    int64_t N = p.get_N();
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

    uint32_t NN = 26;    // default=32, For extreme scenarios, this may need to be increased
       
    //    patch_size = round(p_obj['N'] / patchN)
    double patch_size = std::floor(0.7*(N / p.patch_num) + 0.5);
     
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
        psf *= 0.5 / psf_sum;
        centroid_psf(psf, 0.97);
        
        // # focus_psf, _, _ = centroidPsf(psf, 0.95) : Depending on the size of your PSFs, you may want to use this
        // psf = resize(psf, (round(NN / p_obj['scaling']), round(NN / p_obj['scaling'])))
        cv::resize(psf, psf, cv::Size(std::floor(NN / p.get_scaling() + 0.5), std::floor(NN / p.get_scaling() + 0.5)), 0.0, 0.0, cv::INTER_LINEAR);
        //cv::filter2D(psf, psf, -1, k2, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT(0));
        //cv::GaussianBlur(psf, psf, cv::Size(3, 3), 0);

        // patch_mask = np.zeros((p_obj['N'], p_obj['N']))
        patch_mask = cv::Mat::zeros(N, N, CV_64FC1);

        // patch_mask[round(xx_flat[i]), round(yy_flat[i])] = 1
        // patch_mask = scipy.signal.fftconvolve(patch_mask, np.exp(-patch_indx * *2 / patch_size * *2) * np.exp(-patch_indy * *2 / patch_size * *2) * np.ones((patch_size * 2 + 1, patch_size * 2 + 1)), mode = 'same')
        //patch_mask.at<double>((uint64_t)(*yy.ptr<double>(idx)), (uint64_t)(*xx.ptr<double>(idx))) = 1.0;
        //cv::filter2D(patch_mask, patch_mask, -1, exp_tmp, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT(0));
        
        min_x = std::max(0LL, (int64_t)(*xx.ptr<double>(idx)) - (exp_tmp.cols >> 1) - 1);
        max_x = std::min((int64_t)N, (int64_t)(*xx.ptr<double>(idx)) + (exp_tmp.cols >> 1) + 1);
       
        min_y = std::max(0LL, (int64_t)(*yy.ptr<double>(idx)) - (exp_tmp.rows >> 1) - 1);
        max_y = std::min((int64_t)N, (int64_t)(*yy.ptr<double>(idx)) + (exp_tmp.rows >> 1) + 1);

        min_kx = std::max(0LL, (exp_tmp.cols >> 1) - (int64_t)(*xx.ptr<double>(idx)));
        max_kx = std::min((int64_t)exp_tmp.cols, (int64_t)((exp_tmp.cols >> 1) + (N - (int64_t)(*xx.ptr<double>(idx))) + 1));

        min_ky = std::max(0LL, (exp_tmp.rows >> 1) - (int64_t)(*yy.ptr<double>(idx)));
        max_ky = std::min((int64_t)exp_tmp.rows, (int64_t)((exp_tmp.rows >> 1) + (N - (int64_t)(*yy.ptr<double>(idx))) + 1));

        exp_tmp(cv::Range(min_ky, max_ky), cv::Range(min_kx, max_kx)).copyTo(patch_mask(cv::Range(min_y, max_y), cv::Range(min_x, max_x)));

        // den += scipy.signal.fftconvolve(patch_mask, psf, mode = 'same')
        cv::filter2D(patch_mask, tmp_conv, -1, psf, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT_101);
        den += tmp_conv;
        //cv::Mat tmp_conv2;
        //cv::filter2D(exp_tmp(cv::Range(min_ky, max_ky), cv::Range(min_kx, max_kx)), tmp_conv2, -1, psf, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT(0));
        //cv::Mat den_roi = den(cv::Range(min_y, max_y), cv::Range(min_x, max_x));
        //den_roi += tmp_conv2;

        // img_patches[:, : , i] = scipy.signal.fftconvolve(img * patch_mask, psf, mode = 'same')
        patch_mask = patch_mask.mul(src);
        cv::filter2D(patch_mask, tmp_conv, -1, psf, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT_101);
        img_patches += tmp_conv;

        //cv::Mat t2 = exp_tmp(cv::Range(min_ky, max_ky), cv::Range(min_kx, max_kx)).mul(src(cv::Range(min_y, max_y), cv::Range(min_x, max_x)));

        //cv::filter2D(t2, tmp_conv, -1, psf, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT_101);
        //cv::Mat img_roi = img_patches(cv::Range(min_y, max_y), cv::Range(min_x, max_x));
        //img_roi += tmp_conv;

    }
    // out_img = np.sum(img_patches, axis = 2) / (den + 0.000001)
    dst = img_patches.mul(1.0 / den);

}   // end of generate_blur_image



#endif  // _TEST_H_TURB_SIM_

