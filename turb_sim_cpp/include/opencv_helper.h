#ifndef _OPENCV_HELPER_H_
#define _OPENCV_HELPER_H_


#include <cstdint>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

template <typename T>
//cv::Mat linspace(const T x_start, const T x_stop, const T x_step)
//{
//    std::vector<T> t_x;
//    for (T idx = x_start; idx < x_stop; idx = idx + x_step)
//        t_x.push_back(idx);
//
//    cv::Mat x(t_x);
//
//    return x.reshape(1, 1).clone();
//
//}   // end of linspace

cv::Mat linspace(const T x_start, const T x_stop, uint64_t N)
{
    uint64_t idx;
    std::vector<T> t_x;
    double step = (double)(x_stop - x_start) / (double)(N-1);

    T start = x_start;
    //for (idx = x_start; idx <= x_stop; idx = idx + step)
    for (idx = 0; idx < N; ++idx)
    {
        t_x.push_back(start);
        start += step;
    }
    cv::Mat x(t_x);

    return x.reshape(1, 1).clone();

}   // end of linspace


template <typename T>
void meshgrid(T x_start, T x_stop, uint64_t N_x, T y_start, T y_stop, uint64_t N_y, cv::Mat& X, cv::Mat& Y)
{
    //T idx;

    //std::vector<T> t_x, t_y;
    //for (idx = x_start; idx <= x_stop; idx = idx + x_step)
    //    t_x.push_back(idx);

    //for (idx = y_start; idx <= y_stop; idx = idx + y_step)
    //    t_y.push_back(idx);

    //cv::Mat x(t_x);
    //cv::Mat y(t_y);

    cv::Mat x = linspace(x_start, x_stop, N_x);
    cv::Mat y = linspace(y_start, y_stop, N_y);

    cv::repeat(x, y.total(), 1, X);
    cv::repeat(y.t(), 1, x.total(), Y);
}


void meshgrid(cv::Range xgv, cv::Range ygv, cv::Mat& X, cv::Mat& Y)
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

cv::Mat sqrt_cmplx(cv::Mat& src)
{
    std::complex<double> tmp;
    cv::Mat result = cv::Mat::zeros(src.rows, src.cols, src.type());

    cv::MatIterator_<cv::Vec2d> it, end;
    cv::MatIterator_<cv::Vec2d> res_it = result.begin<cv::Vec2d>();

    for (it = src.begin<cv::Vec2d>(), end = src.end<cv::Vec2d>(); it != end; ++it, ++res_it)
    {
        tmp = std::complex<double>((*it)[0], (*it)[1]);
        tmp = std::sqrt(tmp);

        (*res_it)[0] = tmp.real();
        (*res_it)[1] = tmp.imag();

    }

    return result;
}

cv::Mat abs_cmplx(cv::Mat& src)
{
    std::complex<double> tmp;
    cv::Mat result = cv::Mat::zeros(src.rows, src.cols, CV_64FC1);

    cv::MatIterator_<cv::Vec2d> it, end;
    cv::MatIterator_<double> res_it = result.begin<double>();

    for (it = src.begin<cv::Vec2d>(), end = src.end<cv::Vec2d>(); it != end; ++it, ++res_it)
    {
        tmp = std::complex<double>((*it)[0], (*it)[1]);
        *res_it = std::abs(tmp);
    }

    return result;
}


void threshold_cmplx(cv::Mat& src, cv::Mat &dst, double value)
{
    std::complex<double> tmp;
    cv::Mat result = cv::Mat::zeros(src.rows, src.cols, CV_64FC1);

    cv::MatIterator_<cv::Vec2d> it, end;
    cv::MatIterator_<cv::Vec2d> dst_it = dst.begin<cv::Vec2d>();

    for (it = src.begin<cv::Vec2d>(), end = src.end<cv::Vec2d>(); it != end; ++it, ++dst_it)
    {
        tmp = std::complex<double>((*it)[0], (*it)[1]);
        if (std::abs(tmp) < value)
        {
            (*dst_it)[0] = 0.0;
            (*dst_it)[1] = 0.0;
        }
    }
}


#endif  // _OPENCV_HELPER_H_
