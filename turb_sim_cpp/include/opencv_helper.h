#ifndef _OPENCV_HELPER_H_
#define _OPENCV_HELPER_H_


#include <cstdint>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

template <typename T>
cv::Mat linspace(const T x_start, const T x_stop, const T x_step)
{
    std::vector<T> t_x;
    for (T idx = x_start; idx <= x_stop; idx = idx + x_step)
        t_x.push_back(idx);

    cv::Mat x(t_x);

    return x.reshape(1, 1).clone();

}   // end of linspace

template <typename T>
void meshgrid(const T x_start, const T x_stop, const T x_step, const T y_start, const T y_stop, const T y_step, cv::Mat& X, cv::Mat& Y)
{
    //T idx;

    //std::vector<T> t_x, t_y;
    //for (idx = x_start; idx <= x_stop; idx = idx + x_step)
    //    t_x.push_back(idx);

    //for (idx = y_start; idx <= y_stop; idx = idx + y_step)
    //    t_y.push_back(idx);

    //cv::Mat x(t_x);
    //cv::Mat y(t_y);

    cv::Mat x = linspace(x_start, x_stop, x_step);
    cv::Mat y = linspace(y_start, y_stop, y_step);

    cv::repeat(x, y.total(), 1, X);
    cv::repeat(y.t(), 1, x.total(), Y);
}


void meshgrid(const cv::Range& xgv, const cv::Range& ygv, cv::Mat& X, cv::Mat& Y)
{
    std::vector<int> t_x, t_y;
    for (int i = xgv.start; i <= xgv.end; ++i)
        t_x.push_back(i);

    for (int i = ygv.start; i <= ygv.end; ++i)
        t_y.push_back(i);

    cv::Mat x(t_x);
    cv::Mat y(t_y);

    cv::repeat(x.reshape(1, 1), y.total(), 1, X);
    cv::repeat(y.reshape(1, 1).t(), 1, x.total(), Y);
}


cv::Mat circ(int32_t rows, int32_t cols)
{
    cv::Mat X, Y;
    meshgrid((double)(-rows>>1) + 0.5, (double)(rows >> 1) + 0.5, 1.0, (double)(-cols >> 1) + 0.5, (double)(cols >> 1) + 0.5, 1.0, X, Y);

    cv::Mat r;
    cv::sqrt(cv::abs(X.mul(X)) + cv::abs(Y.mul(Y)), r);
    cv::Mat w = cv::Mat(64, 64, CV_32FC1, cv::Scalar::all(0.0));

    for (int32_t idx = 0; idx < rows; ++idx)
    {
        for (int32_t jdx = 0; jdx < cols; ++jdx)
        {
            if (r.at<double>(idx, jdx) < 31)
                w.at<float>(idx, jdx) = 1.0;
        }
    }

    return w;

}   // end of circ

#endif  // _OPENCV_HELPER_H_
