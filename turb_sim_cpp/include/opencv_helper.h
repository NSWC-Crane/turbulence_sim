#ifndef _OPENCV_HELPER_H_
#define _OPENCV_HELPER_H_


#include <cstdint>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
void meshgrid(cv::Mat x_line, cv::Mat y_line, cv::Mat& X, cv::Mat& Y)
{
    cv::repeat(x_line.reshape(1, 1), y_line.total(), 1, X);
    cv::repeat(y_line.reshape(1, 1).t(), 1, x_line.total(), Y);
}

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
void sqrt_cmplx(cv::Mat& src, cv::Mat &dst)
{
    std::complex<double> tmp;
    //cv::Mat result = cv::Mat::zeros(src.rows, src.cols, src.type());

    cv::MatIterator_<cv::Vec2d> itr, end;
    cv::MatIterator_<cv::Vec2d> dst_it = dst.begin<cv::Vec2d>();

    for (itr = src.begin<cv::Vec2d>(), end = src.end<cv::Vec2d>(); itr != end; ++itr, ++dst_it)
    {
        tmp = std::complex<double>((*itr)[0], (*itr)[1]);
        tmp = std::sqrt(tmp);

        (*dst_it)[0] = tmp.real();
        (*dst_it)[1] = tmp.imag();

    }

    //return result;
}

//-----------------------------------------------------------------------------
cv::Mat abs_cmplx(cv::Mat& src)
{
    std::complex<double> tmp;
    cv::Mat result = cv::Mat::zeros(src.rows, src.cols, CV_64FC1);

    cv::MatIterator_<cv::Vec2d> itr, end;
    cv::MatIterator_<double> res_it = result.begin<double>();

    for (itr = src.begin<cv::Vec2d>(), end = src.end<cv::Vec2d>(); itr != end; ++itr, ++res_it)
    {
        tmp = std::complex<double>((*itr)[0], (*itr)[1]);
        *res_it = std::abs(tmp);
    }

    return result;
}

//-----------------------------------------------------------------------------
void threshold_cmplx(cv::Mat& src, cv::Mat &dst, double value)
{
    double tmp;
    cv::Mat result = cv::Mat::zeros(src.rows, src.cols, CV_64FC1);

    cv::MatIterator_<double> itr, end;
    cv::MatIterator_<cv::Vec2d> dst_it = dst.begin<cv::Vec2d>();

    for (itr = src.begin<double>(), end = src.end<double>(); itr != end; ++itr, ++dst_it)
    {
        //tmp = std::complex<double>((*it)[0], (*it)[1]);
        if (*itr < value)
        {
            (*dst_it)[0] = 0.0;
            (*dst_it)[1] = 0.0;
        }
    }
}

//-----------------------------------------------------------------------------
inline cv::Mat get_channel(cv::Mat& src, uint32_t n)
{
    cv::Mat dst = cv::Mat(src.size(), CV_64FC1);

    cv::MatIterator_<cv::Vec3d> itr;
    cv::MatIterator_<cv::Vec3d> end;
    cv::MatIterator_<double> dst_itr = dst.begin<double>();

    for (itr = src.begin<cv::Vec3d>(), end = src.end<cv::Vec3d>(); itr != end; ++itr, ++dst_itr)
    {
        *dst_itr = (*itr)[n];
    }

    return dst;
}

//-----------------------------------------------------------------------------
inline cv::Mat get_real(cv::Mat& src)
{
    cv::Mat dst = cv::Mat(src.size(), CV_64FC1);

    cv::MatIterator_<cv::Vec2d> itr;
    cv::MatIterator_<cv::Vec2d> end;
    cv::MatIterator_<double> dst_itr = dst.begin<double>();

    for (itr = src.begin<cv::Vec2d>(), end = src.end<cv::Vec2d>(); itr != end; ++itr, ++dst_itr)
    {
        *dst_itr = (*itr)[0];
    }

    return dst;
}

//-----------------------------------------------------------------------------
inline cv::Mat get_imag(cv::Mat& src)
{
    cv::Mat dst = cv::Mat(src.size(), CV_64FC1);

    cv::MatIterator_<cv::Vec2d> itr;
    cv::MatIterator_<cv::Vec2d> end;
    cv::MatIterator_<double> dst_itr = dst.begin<double>();

    for (itr = src.begin<cv::Vec2d>(), end = src.end<cv::Vec2d>(); itr != end; ++itr, ++dst_itr)
    {
        *dst_itr = (*itr)[1];
    }

    return dst;
}

//-----------------------------------------------------------------------------
inline cv::Mat clamp(cv::Mat& src, double min_value, double max_value)
{
    cv::Mat dst = cv::Mat(src.size(), CV_64FC1);

    cv::MatIterator_<double> itr;
    cv::MatIterator_<double> end;
    cv::MatIterator_<double> dst_itr = dst.begin<double>();

    for (itr = src.begin<double>(), end = src.end<double>(); itr != end; ++itr, ++dst_itr)
    {        
        *dst_itr = (*itr < min_value) ? min_value : ((*itr > max_value) ? max_value : *itr);
    }

    return dst;
}

//-----------------------------------------------------------------------------
inline cv::Mat round(cv::Mat& src)
{
    cv::Mat dst = cv::Mat(src.size(), CV_64FC1);

    cv::MatIterator_<double> itr;
    cv::MatIterator_<double> end;
    cv::MatIterator_<double> dst_itr = dst.begin<double>();

    for (itr = src.begin<double>(), end = src.end<double>(); itr != end; ++itr, ++dst_itr)
    {
        *dst_itr = std::floor(*itr + 0.5);
    }

    return dst;
}

//-----------------------------------------------------------------------------
inline cv::Mat cv_atan2(cv::Mat &y_src, cv::Mat& x_src)
{
    cv::Mat dst = cv::Mat(x_src.size(), CV_64FC1);

    cv::MatIterator_<double> x_itr = x_src.begin<double>();
    cv::MatIterator_<double> x_end = x_src.end<double>();
    cv::MatIterator_<double> y_itr = y_src.begin<double>();
    cv::MatIterator_<double> dst_itr = dst.begin<double>();

    for ( ; x_itr != x_end; ++x_itr, ++y_itr, ++dst_itr)
    {
        *dst_itr = std::atan2(*y_itr, *x_itr);
    }

    return dst;
}   // end of cv_atan2

//-----------------------------------------------------------------------------
inline cv::Mat cv_sin(cv::Mat& src)
{
    cv::Mat dst = cv::Mat(src.size(), CV_64FC1);

    cv::MatIterator_<double> itr = src.begin<double>();
    cv::MatIterator_<double> end = src.end<double>();
    cv::MatIterator_<double> dst_itr = dst.begin<double>();

    for (; itr != end; ++itr, ++dst_itr)
    {
        *dst_itr = std::sin(*itr);
    }

    return dst;
}   // end of cv_sin

#endif  // _OPENCV_HELPER_H_
