#ifndef _MOTION_COMPENSTATE_H_
#define _MOTION_COMPENSTATE_H_

#include <cstdint>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "opencv_helper.h"

/*
function g = MotionCompensate(img0, MVx, MVy, pel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Integer pixel motion compensation
%
% g = reconstruct(img0, MVx, MVy, pel)
% constructs a motion compensated frame of img0 according to the motion
% vectors specified by MVx and MVy
%
%
% Stanley Chan
% 29 Apr, 2010
% 10 Feb, 2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img = imresize(img0, 1/pel, 'bilinear');
BlockSize  = floor(size(img,1)/size(MVx,1));
[m n C]    = size(img);
M          = floor(m/block_size)*BlockSize;
N          = floor(n/block_size)*BlockSize;
f          = img(1:M, 1:N, 1:C);
g          = zeros(M, N, C);
MVxmap = imresize(MVx, BlockSize);
MVymap = imresize(MVy, BlockSize);
Dx = round(MVxmap*(1/pel));
Dy = round(MVymap*(1/pel));
[xgrid ygrid] = meshgrid(1:N, 1:M);
X = min(max(xgrid+Dx, 1), N);
Y = min(max(ygrid+Dy, 1), N);
idx = (X(:)-1)*M + Y(:);
for coloridx = 1:C
    fc = f(:,:,coloridx);
    g(:,:,coloridx) = reshape(fc(idx), M, N);
end
g = imresize(g, pel);
*/

/*
def motion_compensate(img, Mvx, Mvy, pel):
    m, n =  np.shape(img)[0], np.shape(img)[1]
    img = resize(img, (np.int32(m/pel), np.int32(n/pel)), mode = 'reflect' )
    Blocksize = np.floor(np.shape(img)[0]/np.shape(Mvx)[0])
    m, n =  np.shape(img)[0], np.shape(img)[1]
    M, N =  np.int32(np.ceil(m/Blocksize)*Blocksize), np.int32(np.ceil(n/Blocksize)*Blocksize)

    f = img[0:M, 0:N]


    Mvxmap = resize(Mvy, (N,M))
    Mvymap = resize(Mvx, (N,M))


    xgrid, ygrid = np.meshgrid(np.arange(0,N-0.99), np.arange(0,M-0.99))
    X = np.clip(xgrid+np.round(Mvxmap/pel),0,N-1)
    Y = np.clip(ygrid+np.round(Mvymap/pel),0,M-1)

    idx = np.int32(Y.flatten()*N + X.flatten())
    f_vec = f.flatten()
    g = np.reshape(f_vec[idx],[N,M])

    g = resize(g, (np.shape(g)[0]*pel,np.shape(g)[1]*pel))
    return g
*/


void motion_compensate(cv::Mat &src, cv::Mat &dst, cv::Mat &mv_x, cv::Mat &mv_y, double pel)
{
    uint64_t idx;
    cv::Mat img;
    cv::Mat mv_xmap, mv_ymap;
    uint32_t img_h = src.rows;
    uint32_t img_w = src.cols;

    cv::resize(src, img, cv::Size(), (uint32_t)(img_w/pel), (uint32_t)(img_h /pel), cv::INTER_LINEAR);

    // BlockSize  = floor(size(img,1)/size(MVx,1));
    uint32_t block_size = std::floor(img_h / (double)mv_x.rows);

    // M          = floor(m/block_size)*BlockSize;
    // N          = floor(n/block_size)*BlockSize;
    uint32_t M = (uint32_t)(std::ceil(img_h / (double)block_size) * block_size);
    uint32_t N = (uint32_t)(std::ceil(img_w / (double)block_size) * block_size);

    img_h = img.rows;
    img_w = img.cols;

    // f          = img(1:M, 1:N, 1:C);
    cv::Mat f(img, cv::Rect(0, 0, M, N));

    cv::resize(mv_x, mv_xmap, cv::Size(M, N));
    cv::resize(mv_y, mv_ymap, cv::Size(M, N));

    // xgrid, ygrid = np.meshgrid(np.arange(0,N-0.99), np.arange(0,M-0.99))
    cv::Mat x_grid, y_grid;
    meshgrid<double>(0, N-1, 1, 0, M-1, 1, x_grid, y_grid);
    x_grid += (mv_xmap * (1.0 / pel));
    y_grid += (mv_ymap * (1.0 / pel));

    // X = np.clip(xgrid + np.round(Mvxmap / pel), 0, N - 1)
    // Y = np.clip(ygrid + np.round(Mvymap / pel), 0, M - 1)
    cv::Mat X = clamp(x_grid, 0, N - 1.0);
    cv::Mat Y = clamp(y_grid, 0, N - 1.0);

    // idx = np.int32(Y.flatten()*N + X.flatten())
    //std::vector<uint32_t> index(M * N);
    uint32_t index;
    cv::Mat g = cv::Mat::zeros(M, N, CV_64FC1);
    cv::MatIterator_<double> X_itr = X.begin<double>();
    cv::MatIterator_<double> X_end = X.end<double>();
    cv::MatIterator_<double> Y_itr = Y.begin<double>();
    cv::MatIterator_<double> f_itr = f.begin<double>();
    cv::MatIterator_<double> g_itr = g.begin<double>();

    for ( ; X_itr != X_end; ++X_itr, ++Y_itr, ++g_itr)
    {
        index = (uint32_t)((*Y_itr) * N + (*X_itr));
        *g_itr = *(f_itr + index);
    }

    // f_vec = f.flatten()
    // g = np.reshape(f_vec[idx],[N,M])
    // g = resize(g, (np.shape(g)[0] * pel, np.shape(g)[1] * pel))
    cv::resize(g, dst, cv::Size((uint32_t)(M * pel), (uint32_t)(N * pel)), cv::INTER_LINEAR);

}   // end of motion_compensate


#endif  // _MOTION_COMPENSTATE_H_
