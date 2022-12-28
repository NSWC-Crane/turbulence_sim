#ifndef _MOTION_COMPENSTATE_H_
#define _MOTION_COMPENSTATE_H_

#include <cstdint>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "opencv_helper.h"

//-----------------------------------------------------------------------------
// Warping Function for Turbulence Simulator
//
// C++ version of original code by Stanley Chan
// 
// adapted from here:
// https://github.itap.purdue.edu/StanleyChanGroup/TurbulenceSim_v1/blob/master/Turbulence_Sim_v1_python/Motion_Compensate.py
//
void motion_compensate(cv::Mat &src, cv::Mat &dst, cv::Mat &mv_x, cv::Mat &mv_y, double pel)
{
    uint64_t idx;
    cv::Mat img;
    cv::Mat mv_xmap, mv_ymap;
    uint32_t img_h, img_w, img_ch;
    uint32_t index;

    // img = resize(img, (np.int32(m/pel), np.int32(n/pel)), mode = 'reflect' )
    cv::resize(src, img, cv::Size((uint32_t)(src.cols /pel), (uint32_t)(src.rows /pel)), 0, 0, cv::INTER_LINEAR);

    img_h = img.rows;
    img_w = img.cols;
    img_ch = img.channels();

    // BlockSize  = floor(size(img,1)/size(MVx,1));
    uint32_t block_size = std::floor(img_h / (double)mv_x.rows);

    // M = floor(m/block_size)*BlockSize;
    // N = floor(n/block_size)*BlockSize;
    uint32_t M = (uint32_t)(std::ceil(img_w / (double)block_size) * block_size);
    uint32_t N = (uint32_t)(std::ceil(img_h / (double)block_size) * block_size);


    cv::resize(mv_x, mv_xmap, cv::Size(M, N), 0, 0, cv::INTER_LINEAR);
    cv::resize(mv_y, mv_ymap, cv::Size(M, N), 0, 0, cv::INTER_LINEAR);

    // xgrid, ygrid = np.meshgrid(np.arange(0,N-0.99), np.arange(0,M-0.99))
    cv::Mat x_grid, y_grid;
    meshgrid<double>(0, M-1, N, 0, N-1, M, x_grid, y_grid);
    mv_xmap *= (1.0 / pel);
    mv_ymap *= (1.0 / pel);

    x_grid += round(mv_xmap);
    y_grid += round(mv_ymap);

    // X = np.clip(xgrid + np.round(Mvxmap / pel), 0, N - 1)
    // Y = np.clip(ygrid + np.round(Mvymap / pel), 0, M - 1)
    cv::Mat X = clamp(x_grid, 0, M - 1);
    cv::Mat Y = clamp(y_grid, 0, N - 1);

    // idx = np.int32(Y.flatten()*N + X.flatten())
    //std::vector<uint32_t> index(M * N);

    // f = img(1:M, 1:N, 1:C);
    cv::Mat f = img(cv::Rect(0, 0, M, N)).reshape(1, 1);
    cv::Mat g = cv::Mat::zeros(f.rows, f.cols, f.type());


    cv::MatIterator_<double> X_itr = X.begin<double>();
    cv::MatIterator_<double> X_end = X.end<double>();
    cv::MatIterator_<double> Y_itr = Y.begin<double>();
    cv::MatIterator_<double> f_itr = f.begin<double>();
    cv::MatIterator_<double> g_itr = g.begin<double>();

    for ( ; X_itr != X_end; ++X_itr, ++Y_itr)
    {
        index = (uint32_t)((*Y_itr) * N + (*X_itr));

        for (idx = 0; idx < img_ch; ++idx)
        {
            *(g_itr+idx) = *(f_itr + idx + (img_ch * index));
            ++g_itr;
        }
    }

    // g = np.reshape(f_vec[idx],[N,M])
    // g = resize(g, (np.shape(g)[0] * pel, np.shape(g)[1] * pel))
    g = g.reshape(img.channels(), N);
    cv::resize(g, dst, cv::Size((uint32_t)(M * pel), (uint32_t)(N * pel)), cv::INTER_LINEAR);

}   // end of motion_compensate


#endif  // _MOTION_COMPENSTATE_H_
