#define _CRT_SECURE_NO_WARNINGS

#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
#include <windows.h>

#else
#include <dlfcn.h>
typedef void* HINSTANCE;

#endif

// C/C++ includes
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <type_traits>
#include <list>
#include <set>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgcodecs.hpp>

// custom includes
#include <num2string.h>
#include <file_ops.h>

#include "cv_dft_conv.h"
#include "motion_compensate.h"


// ----------------------------------------------------------------------------------------
//bool compare(std::pair<uint8_t, uint8_t> p1, std::pair<uint8_t, uint8_t> p2)
//{
//    return std::max(p1.first, p1.second) < std::max(p2.first, p2.second);
//}

// ----------------------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& out, std::vector<uint8_t>& item)
{
    for (uint64_t idx = 0; idx < item.size() - 1; ++idx)
    {
        out << static_cast<uint32_t>(item[idx]) << ",";
    }
    out << static_cast<uint32_t>(item[item.size() - 1]);
    return out;
}

// ----------------------------------------------------------------------------------------
template <typename T>
inline std::ostream& operator<<(std::ostream& out, std::vector<T>& item)
{
    for (uint64_t idx = 0; idx < item.size() - 1; ++idx)
    {
        out << item[idx] << ",";
    }
    out << item[item.size() - 1];
    return out;
}


static double bessj0(double x)
/*------------------------------------------------------------*/
/* PURPOSE: Evaluate Bessel function of first kind and order  */
/*          0 at input x                                      */
/*------------------------------------------------------------*/
{
    double ax, z;
    double xx, y, ans, ans1, ans2;

    if ((ax = fabs(x)) < 8.0) {
        y = x * x;
        ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7
            + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718
            + y * (59272.64853 + y * (267.8532712 + y * 1.0))));
        ans = ans1 / ans2;
    }
    else {
        z = 8.0 / ax;
        y = z * z;
        xx = ax - 0.785398164;
        ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
            + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        ans2 = -0.1562499995e-1 + y * (0.1430488765e-3
            + y * (-0.6911147651e-5 + y * (0.7621095161e-6
                - y * 0.934935152e-7)));
        ans = sqrt(0.636619772 / ax) * (cos(xx) * ans1 - z * sin(xx) * ans2);
    }
    return ans;
}


double BESSJ0(double X) {
    /***********************************************************************
          This subroutine calculates the First Kind Bessel Function of
          order 0, for any real number X. The polynomial approximation by
          series of Chebyshev polynomials is used for 0<X<8 and 0<8/X<1.
          REFERENCES:
          M.ABRAMOWITZ,I.A.STEGUN, HANDBOOK OF MATHEMATICAL FUNCTIONS, 1965.
          C.W.CLENSHAW, NATIONAL PHYSICAL LABORATORY MATHEMATICAL TABLES,
          VOL.5, 1962.
    ************************************************************************/
    const double
        P1 = 1.0, P2 = -0.1098628627E-2, P3 = 0.2734510407E-4,
        P4 = -0.2073370639E-5, P5 = 0.2093887211E-6,
        Q1 = -0.1562499995E-1, Q2 = 0.1430488765E-3, Q3 = -0.6911147651E-5,
        Q4 = 0.7621095161E-6, Q5 = -0.9349451520E-7,
        R1 = 57568490574.0, R2 = -13362590354.0, R3 = 651619640.7,
        R4 = -11214424.18, R5 = 77392.33017, R6 = -184.9052456,
        S1 = 57568490411.0, S2 = 1029532985.0, S3 = 9494680.718,
        S4 = 59272.64853, S5 = 267.8532712, S6 = 1.0;
    double
        AX, FR, FS, Z, FP, FQ, XX, Y, TMP;

    if (X == 0.0) return 1.0;
    AX = fabs(X);
    if (AX < 8.0) {
        Y = X * X;
        FR = R1 + Y * (R2 + Y * (R3 + Y * (R4 + Y * (R5 + Y * R6))));
        FS = S1 + Y * (S2 + Y * (S3 + Y * (S4 + Y * (S5 + Y * S6))));
        TMP = FR / FS;
    }
    else {
        Z = 8. / AX;
        Y = Z * Z;
        XX = AX - 0.785398164;
        FP = P1 + Y * (P2 + Y * (P3 + Y * (P4 + Y * P5)));
        FQ = Q1 + Y * (Q2 + Y * (Q3 + Y * (Q4 + Y * Q5)));
        TMP = sqrt(0.636619772 / AX) * (FP * cos(XX) - Z * FQ * sin(XX));
    }
    return TMP;
}

double Sign(double X, double Y) {
    if (Y < 0.0) return (-fabs(X));
    else return (fabs(X));
}

// ----------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int bp = 0;

    uint32_t idx = 0, jdx = 0;
    uint32_t img_h = 512;
    uint32_t img_w = 512;
    cv::Size img_size(img_h, img_w);

    cv::RNG rng(time(NULL));

    // timing variables
    typedef std::chrono::duration<double> d_sec;
    auto start_time = std::chrono::system_clock::now();
    auto stop_time = std::chrono::system_clock::now();
    auto elapsed_time = std::chrono::duration_cast<d_sec>(stop_time - start_time);

    cv::Mat img_f1, img_f2;

    std::string window_name = "montage";



    //if (argc == 1)
    //{
    //    std::cout << "Error: Missing confige file" << std::endl;
    //    std::cout << "Usage: ./pg <confige_file.txt>" << std::endl;
    //    std::cout << std::endl;
    //    std::cin.ignore();
    //    return 0;
    //}

    // setup the windows to display the results
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 2*img_w, img_h);

    // do work here
    try
    {    
        const cv::Range xgv = cv::Range(1, 3);
        const cv::Range ygv = cv::Range(0, 5);
        cv::Mat Y;
        //meshgrid(xgv, ygv, X, Y);

        cv::Mat X = linspace(-31.5, 31.5, 1.0);
        double b0 = _j0(2.1);


        double b1 = bessj0(2.1);
        double b2 = BESSJ0(2.1);

        meshgrid(-31.5, 31.5, 1.0, -31.5, 31.5, 1.0, X, Y);

        cv::Mat r;
        cv::sqrt(cv::abs(X.mul(X)) + cv::abs(Y.mul(Y)), r);
        cv::Mat circ = cv::Mat(64, 64, CV_32FC1, cv::Scalar::all(0.0));

        for (idx = 0; idx < 64; ++idx)
        {
            for (jdx = 0; jdx < 64; ++jdx)
            {
                if (r.at<double>(idx, jdx) < 31)
                    circ.at<float>(idx, jdx) = 1.0;
            }
        }

        
        //cv::circle(circ, cv::Point(31, 31), 31, 255, 0, cv::LineTypes::LINE_8, 0);


        bp = 1;

    }
    catch(std::exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
    }


    std::cout << "End of Program.  Press Enter to close..." << std::endl;
	std::cin.ignore();
    cv::destroyAllWindows();

}   // end of main

