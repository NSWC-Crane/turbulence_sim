#ifndef _ZERNIKE_FUNCTIONS_H_
#define _ZERNIKE_FUNCTIONS_H_

#include <cstdint>
#include <cmath>
#include <vector>
#include <iostream>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Custom includes
#include "turbulence_param.h"
#include "turbulence_sim.h"
#include "opencv_helper.h"
#include "noll_functions.h"


//-----------------------------------------------------------------------------
inline void centroid_psf(cv::Mat& psf, double threshold = 0.95)
{
    int32_t radius = 4;
    int32_t min_x, max_x;
    int32_t min_y, max_y;

    double temp_sum = 0.0;
    cv::Mat mx, my;
    cv::Mat tmp_psf;

    cv::Point center;

    cv::minMaxLoc(psf, NULL, NULL, NULL, &center);

    //x = np.linspace(0, psf.shape[0], psf.shape[0])
    //y = np.linspace(0, psf.shape[1], psf.shape[1])
    //col, row = np.meshgrid(x, y)
    //meshgrid<double>(0, psf.cols - 1, psf.cols, 0, psf.rows - 1, psf.rows, mx, my);

    try
    {
        //center.x = (uint8_t)cv::sum(mx.mul(psf))[0];
        //center.y = (uint8_t)cv::sum(my.mul(psf))[0];

        while (temp_sum < threshold)
        {
            ++radius;

            min_x = std::max(0, center.x - radius);
            max_x = std::min(psf.cols, center.x + radius + 1);
            min_y = std::max(0, center.y - radius);
            max_y = std::min(psf.rows, center.y + radius + 1);

            // make sure that this thing doesn't run away
            if ((max_x - min_x) == psf.cols || (max_y - min_y) == psf.rows)
                break;

            // return_psf = psf[cen_row - radius:cen_row + radius + 1, cen_col - radius : cen_col + radius + 1]
            tmp_psf = psf(cv::Range(min_y, max_y), cv::Range(min_x, max_x));
            temp_sum = cv::sum(tmp_psf)[0];
        }

        psf = tmp_psf.clone();
    }
    catch (std::exception e)
    {
        std::cout << "error: " << e.what() << std::endl;
        std::cout << "Filename: " << __FILE__ << std::endl;
        std::cout << "Line #: " << __LINE__ << std::endl;
    }
}   // end of centroid_psf

//-----------------------------------------------------------------------------
//Just a simple function to generate random coefficients as needed, conforms to Zernike's Theory. The nollCovMat()
//function is at the heart of this function.

//A note about the function call of nollCovMat in this function. The input (..., 1, 1) is done for the sake of
//flexibility. One can call the function in the typical way as is stated in its description. However, for
//generality, the D/r0 weighting is pushed to the "b" random vector, as the covariance matrix is merely scaled by
//such value.

//:param num_zern: This is the number of Zernike basis functions/coefficients used. Should be numbers that the pyramid
//rows end at. For example [1, 3, 6, 10, 15, 21, 28, 36]
//:param D_r0:
//
void generate_zernike_coeff(uint32_t N, double v, std::vector<double> &coeff, cv::RNG &rng)
{
    double c1 = std::exp((3.0 / 10.0) * std::log(v));

    coeff.clear();

    // C = nollCovMat(N, 1, 1)
    cv::Mat ncm = noll_covariance_matrix(N, 1, 1);

    // e_val, e_vec = np.linalg.eig(C)
    cv::Mat e_val, e_vec;
    eigen(ncm, e_val, e_vec);

    // R = np.real(e_vec * np.sqrt(e_val))
    cv::sqrt(e_val, e_val);
    e_val = cv::Mat::diag(e_val);
    cv::Mat cov_mat = e_vec * e_val;
    //R = get_real(R);

    // b = np.random.randn(int(num_zern), 1) * v ** (3.0/10.0)
    cv::Mat b(N, 1, CV_64FC1);
    rng.fill(b, cv::RNG::NORMAL, 0.0, 1.0);
    b *= (c1 * c1);

    // a = np.matmul(R, b)
    //cv::Mat dst = R * b;
    cv::Mat a = cov_mat * b;

    cv::MatIterator_<double> itr;
    cv::MatIterator_<double> end;

    for (itr = a.begin<double>(), end = a.end<double>(); itr != end; ++itr)
    {
        coeff.push_back(*itr);
    }

}   // end of generate_zernike_coeff

//-----------------------------------------------------------------------------
cv::Mat radial_zernike(int64_t n, int64_t m, cv::Mat& x_grid, cv::Mat& y_grid)
{
    uint64_t idx;
    double tmp;
    cv::Mat xx = x_grid.mul(x_grid);
    cv::Mat yy = y_grid.mul(y_grid);
    cv::Mat rho = xx + yy;
    cv::Mat tmp_pow;
    double t1, t2, t3, t4;

    //rho = np.sqrt(x_grid * *2 + y_grid * *2)
    cv::sqrt(rho, rho);

    //radial = np.zeros(rho.shape)
    cv::Mat radial = cv::Mat::zeros(rho.size(), CV_64FC1);

    m = std::abs(m);

    //for k in range(int((n - m) / 2 + 1)) :
    for (idx = 0; idx < (uint64_t)(((n - m) / 2.0) + 1); ++idx)
    {
        t1 = ((idx & 0x01 == 1) ? -1.0 : 1.0);
        t2 = std::tgamma(n - idx + 1);
        t3 = (std::tgamma(idx + 1) * std::tgamma(((n + m) / 2.0) - idx + 1) * std::tgamma(((n - m) / 2.0) - idx + 1));
        // temp = (-1) * *k * np.math.factorial(n - k) / (np.math.factorial(k) * np.math.factorial((n + m) / 2 - k) * np.math.factorial((n - m) / 2 - k))
        tmp = ((idx & 0x01 == 1) ? -1.0 : 1.0) * std::tgamma(n - idx + 1) / (std::tgamma(idx + 1) * std::tgamma(((n + m) / 2.0) - idx + 1) * std::tgamma(((n - m) / 2.0) - idx + 1));

        //    radial += temp * rho * *(n - 2 * k)
        cv::pow(rho, (n - 2 * idx), tmp_pow);
        radial += (tmp * tmp_pow);
    }

    return radial;

}   // end of radial_zernike



//-----------------------------------------------------------------------------
// This function simply
// : param index :
// : param x_grid :
// : param y_grid :
//
cv::Mat generate_zernike_poly(int64_t N, cv::Mat& x_grid, cv::Mat& y_grid)
{
    int64_t n, m;
    cv::Mat tmp = cv::Mat::zeros(x_grid.size(), CV_64FC1);

    noll2zernike_index(N, n, m);

    //radial = radialZernike(x_grid, y_grid, (n, m))
    cv::Mat radial = radial_zernike(n, m, x_grid, y_grid);
    cv::Mat dst;

    cv::MatIterator_<double> x_itr = x_grid.begin<double>();
    cv::MatIterator_<double> x_end = x_grid.end<double>();
    cv::MatIterator_<double> y_itr = y_grid.begin<double>();
    cv::MatIterator_<double> tmp_itr = tmp.begin<double>();

    if (m < 0)
    {
        // return np.multiply(radial, np.sin(-m * np.arctan2(y_grid, x_grid)))
        for (; x_itr != x_end; ++x_itr, ++y_itr, ++tmp_itr)
        {
            *tmp_itr = std::sin((-m) * std::atan2(*y_itr, *x_itr));
        }
    }
    else
    {
        // return np.multiply(radial, np.cos(m * np.arctan2(y_grid, x_grid)))
        for (; x_itr != x_end; ++x_itr, ++y_itr, ++tmp_itr)
        {
            *tmp_itr = std::cos(m * std::atan2(*y_itr, *x_itr));
        }
    }

    dst = radial.mul(tmp);
    return dst;
}   // generate_zernike_poly


//-----------------------------------------------------------------------------
// Generating the Zernike Phase representation.
// This implementation uses Noll's indices. 1 -> (0,0), 2 -> (1,1), 3 -> (1, -1), 4 -> (2,0), 5 -> (2, -2), etc.
//def zernikeGen(N, coeff, **kwargs)
// 
//void generate_zernike_phase(int64_t N, std::vector<double>& coeff, cv::Mat& zern_out, cv::Mat &x_grid, cv::Mat &y_grid)
void generate_zernike_phase(int64_t N, cv::Mat& coeff, cv::Mat& zern_out, cv::Mat& x_grid, cv::Mat& y_grid)
{
    uint64_t idx = 0;
    //cv::Mat x_grid, y_grid;

    //num_coeff = coeff.size
    uint64_t num_coeff = coeff.total();

    // Setting up 2D grid
    //x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, N, endpoint = True), np.linspace(-1, 1, N, endpoint = True))
    //meshgrid(-1.0, 1.0, N, -1.0, 1.0, N, x_grid, y_grid);

    //#mask = np.sqrt(x_grid * *2 + y_grid * *2) <= 1
    //#x_grid = x_grid * mask
    //#y_grid = y_grid * mask

    //zern_out = np.zeros((N, N, num_coeff))
    zern_out = cv::Mat::zeros(N, N, CV_64FC1);

    //for (idx = 0; idx < num_coeff; ++idx)
    //{
    //    // zern_out[:, : , i] = coeff[i] * genZernPoly(i + 1, x_grid, y_grid)
    //    zern_out += coeff[idx] * generate_zernike_poly(idx + 1, x_grid, y_grid);
    //}

    cv::MatIterator_<double> c_itr = coeff.begin<double>();
    cv::MatIterator_<double> c_end = coeff.end<double>();
    for (; c_itr != c_end; ++c_itr, ++idx)
    {
        zern_out += (*c_itr) * generate_zernike_poly(idx + 1, x_grid, y_grid);
    }

}   // end of generate_zernike_phase


//-----------------------------------------------------------------------------
//def psfGen(N, **kwargs) :
//    EXAMPLE USAGE - TESTING IN MAIN
//    temp, xx, yy, pupil = util.psfGen(256, pad_size = 1024)
//    psf = np.abs(temp) * *2
//    print(np.min(xx.ravel()), np.max(xx.ravel()))
//    plt.imshow(psf / np.max(psf.ravel()), extent = [np.min(xx.ravel()), np.max(xx.ravel()),
//        np.min(yy.ravel()), np.max(yy.ravel())])
//    plt.show()
//
//    :param N :
//: param kwargs :
// generate_psf(NN, coeff=aa, L=p_obj['L'], D=p_obj['D'], z_i=1.2, wavelength=p_obj['wvl'])
void generate_psf(uint64_t N, turbulence_param &p, cv::RNG& rng, cv::Mat& psf, double z_i = 1.2, double pad_size = 0.0)
{
    
    cv::Mat x_grid, x_grid2, y_grid, y_grid2;
    cv::Mat x_samp_grid, y_samp_grid;
    cv::Mat mask;
    std::complex<double> j(0, 1);
    std::vector<double> coeff;
    double psf_sum;

    //    wavelength = kwargs.get('wavelength', 500 * (10 * *(-9)))
    //    pad_size = kwargs.get('pad_size', 0)
    //    D = kwargs.get('D', 0.1)
    //    L = kwargs.get('L', -1)
    //    z_i = kwargs.get('z_i', 1.2)
    //    vec = kwargs.get('coeff', np.asarray([[1], [0], [0], [0], [0], [0], [0], [0], [0]] ))
     
    // b = np.random.randn(int(num_zern), 1) * v ** (3.0/10.0)
    cv::Mat b(p.num_zern_coeff, 1, CV_64FC1);
    rng.fill(b, cv::RNG::NORMAL, 0.0, 1.0);
    b *= (p.cp[0].zern_c1);

    // a = np.matmul(R, b)
    //cv::Mat dst = R * b;
    cv::Mat a = p.cov_mat * b;

    cv::MatIterator_<double> itr;
    cv::MatIterator_<double> end;

    //for (itr = a.begin<double>(), end = a.end<double>(); itr != end; ++itr)
    //{
    //    coeff.push_back(*itr);
    //}
     
    //    x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, N, endpoint = True), np.linspace(-1, 1, N, endpoint = True))
    meshgrid(-1.0, 1.0, N, -1.0, 1.0, N, x_grid, y_grid);

    //    mask = np.sqrt(x_grid * *2 + y_grid * *2) <= 1
    x_grid2 = x_grid.mul(x_grid);
    y_grid2 = y_grid.mul(y_grid);
    mask = x_grid2 + y_grid2;

    cv::sqrt(mask, mask);
    //mask = (mask <= 1);
    cv::threshold(mask, mask, 1.0, 1.0, cv::THRESH_BINARY_INV);

    //    zernike_stack = zernikeGen(N, vec)
    //    phase = np.sum(zernike_stack, axis = 2)
    std::vector<cv::Mat> zernike_stack;
    cv::Mat phase;
    generate_zernike_phase(N, a, phase, x_grid, y_grid);
        
    //    wave = np.exp((1j * 2 * np.pi * phase)) * mask
    cv::Mat wave = exp_cmplx(2 * CV_PI * j, phase);
    wave = mul_cmplx(mask, wave);
    //phase *= mask;

    //    pad_wave = np.pad(wave, int(pad_size / 2), 'constant', constant_values = 0)
    
    //    #c_psf = np.fft.fftshift(np.fft.fft2(pad_wave))
    //    h = np.fft.fftshift(np.fft.ifft2(pad_wave))
    //cv::Mat h;
    cv::dft(wave, psf, cv::DFT_INVERSE + cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE, wave.rows);
    fftshift(psf);

    //psf = np.abs(temp) * *2
    psf = abs_cmplx(psf);
    psf = psf.mul(psf);

    // psf = psf / np.sum(psf.ravel())
    psf_sum = cv::sum(psf)[0];
    psf *= 1.0 / psf_sum;
    
    //    #pad_wave = np.abs(pad_wave) * *2
    //    # numpy.correlate(x, x, mode = 'same')
    //    #plt.imshow(phase * mask)
    //    #plt.show()
    //    M = pad_size + N
       
    //    fs = N * wavelength * z_i / D
    //double fs = 0.5 * N * p.get_wavelength() * (z_i / p.get_D());
    
    //    temp = np.linspace(-fs / 2, fs / 2, M)  
    //    x_samp_grid, y_samp_grid = np.meshgrid(temp, -temp)
    //meshgrid(-fs, fs, pad_size + N, fs, -fs, pad_size + N, x_samp_grid, y_samp_grid);
    //x_samp_grid *= (p.get_L() / z_i);
    //y_samp_grid *= (p.get_L() / z_i);

    //    if L == -1:
    //return h, x_samp_grid, y_samp_grid, phase* mask,
    //    else:
    //return h, (L / z_i)* x_samp_grid, (L / z_i)* y_samp_grid, phase* mask, wave

}   // end of generate_psf

//-----------------------------------------------------------------------------
void generate_rgb_psf(uint64_t N, turbulence_param& p, cv::RNG& rng, cv::Mat &psf, double z_i = 1.2, double pad_size = 0.0)
//void generate_rgb_psf(uint64_t N, turbulence_param& p, cv::RNG& rng, std::vector<cv::Mat> &psf_v, double z_i = 1.2, double pad_size = 0.0)
{
    uint32_t idx, jdx;
    cv::Mat x_grid, x_grid2, y_grid, y_grid2;
    cv::Mat x_samp_grid, y_samp_grid;
    cv::Mat mask;
    cv::Mat b2;
    cv::Mat a;
    cv::Mat phase;
    cv::Mat tmp_psf;
    std::complex<double> j(0, 1);
    std::vector<double> coeff(p.num_zern_coeff);
    std::vector<cv::Mat> psf_v(p.cp.size());
    //psf_v.resize(p.cp.size());
    double psf_sum = 0.0;
    int32_t top, left, bot, right;

    //    x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, N, endpoint = True), np.linspace(-1, 1, N, endpoint = True))
    meshgrid(-1.0, 1.0, N, -1.0, 1.0, N, x_grid, y_grid);

    //    mask = np.sqrt(x_grid * *2 + y_grid * *2) <= 1
    x_grid2 = x_grid.mul(x_grid);
    y_grid2 = y_grid.mul(y_grid);
    mask = x_grid2 + y_grid2;

    cv::sqrt(mask, mask);
    //mask = (mask <= 1);
    cv::threshold(mask, mask, 1.0, 1.0, cv::THRESH_BINARY_INV);

    // b = np.random.randn(int(num_zern), 1) * v ** (3.0/10.0)
    cv::Mat b(p.num_zern_coeff, 1, CV_64FC1);
    rng.fill(b, cv::RNG::NORMAL, 0.0, 1.0);

    cv::MatIterator_<double> itr;
    cv::MatIterator_<double> end;

    int32_t max_width = 0, max_height = 0;

    for (idx = 0; idx < p.cp.size(); ++idx)
    {

        b2 = (p.cp[idx].zern_c1) * b;

        // a = np.matmul(R, b)
        a = p.cov_mat * b2;
        
        //jdx = 0;
        //for (itr = a.begin<double>(), end = a.end<double>(); itr != end; ++itr)
        //{
        //    coeff[jdx++] = (*itr);
        //}

        generate_zernike_phase(N, a, phase, x_grid, y_grid);

        //    wave = np.exp((1j * 2 * np.pi * phase)) * mask
        cv::Mat wave = exp_cmplx(0.75*2 * CV_PI * j, phase);
        wave = mul_cmplx(mask, wave);

        cv::dft(wave, tmp_psf, cv::DFT_INVERSE + cv::DFT_COMPLEX_OUTPUT + cv::DFT_SCALE, wave.rows);
        fftshift(tmp_psf);

        //psf = np.abs(temp) * *2
        psf_v[idx] = abs_cmplx(tmp_psf);
        psf_v[idx] = psf_v[idx].mul(psf_v[idx]);

        // psf = psf / np.sum(psf.ravel())
        psf_sum = cv::sum(psf_v[idx])[0];
        psf_v[idx] *= 1.0 / psf_sum;

        centroid_psf(psf_v[idx], 0.98);

        max_width = std::max(psf_v[idx].cols, max_width);
        max_height = std::max(psf_v[idx].rows, max_height);
    }

    // make the psf size identical
    for (idx = 0; idx < p.cp.size(); ++idx)
    {
        top = (max_height - psf_v[idx].rows) >> 1;
        bot = max_height - psf_v[idx].rows - top;
        left = (max_width - psf_v[idx].cols) >> 1;
        right = max_width - psf_v[idx].cols - left;
        cv::copyMakeBorder(psf_v[idx], psf_v[idx], top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(0));
    }

    cv::merge(psf_v, psf);

}   // end of generate_rgb_psf

#endif  // _ZERNIKE_FUNCTIONS_H_