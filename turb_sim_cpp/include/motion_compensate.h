#ifndef _MOTION_COMPENSTATE_H_
#define _MOTION_COMPENSTATE_H_

#include <cstdint>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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

cv::Mat motion_compensate(cv::Mat &src, cv::Mat MVx, cv::Mat MVy, double pel)
{
    cv::Mat img;
    cv::resize(src, img, cv::Size(), 1.0/pel, 1.0/pel, cv::INTER_LINEAR);

    uint32_t img_h = img.rows;
    uint32_t img_w = img.cols;

    // BlockSize  = floor(size(img,1)/size(MVx,1));
    int32_t block_size = std::floor(img_h / (double)MVx.rows);

    // M          = floor(m/block_size)*BlockSize;
    // N          = floor(n/block_size)*BlockSize;
    int32_t M = std::floor(img_h / (double)block_size) * block_size;
    int32_t N = std::floor(img_w / (double)block_size) * block_size;

    // f          = img(1:M, 1:N, 1:C);
    cv::Mat f(img, cv::Rect(0, 0, N, M));

    // g          = zeros(M, N, C);
    cv::Mat g = cv::Mat::zeros(M, N, src.type());

    // MVxmap = imresize(MVx, BlockSize);
    // MVymap = imresize(MVy, BlockSize);
    // Dx = round(MVxmap*(1/pel));
    // Dy = round(MVymap*(1/pel));
    
    // [xgrid ygrid] = meshgrid(1:N, 1:M);
    cv::Mat x_grid, y_grid;
    //meshgrid(cv::Range(0, N-1), cv::Range(0, M-1), x_grid, y_grid);
    
    // X = min(max(xgrid+Dx, 1), N);
    // Y = min(max(ygrid+Dy, 1), N);
    
    // idx = (X(:)-1)*M + Y(:);
    
    // for coloridx = 1:C
        // fc = f(:,:,coloridx);
        // g(:,:,coloridx) = reshape(fc(idx), M, N);
    // end
    
    // g = imresize(g, pel);
    
    
    return img;

}   // end of motion_compensate


#endif  // _MOTION_COMPENSTATE_H_
