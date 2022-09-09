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
#include "turbulence_sim.h"

#include <gsl/gsl_linalg.h>

// https://www.atnf.csiro.au/computing/software/gipsy/sub/bessel.c
#define ACC 40.0
#define BIGNO 1.0e10
#define BIGNI 1.0e-10

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
//------------------------------------------------------------
// PURPOSE: Evaluate Bessel function of first kind and order  
//          0 at input x   
// https://www.atnf.csiro.au/computing/software/gipsy/sub/bessel.c                                   
//------------------------------------------------------------
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

static double bessj1( double x )
//------------------------------------------------------------
// PURPOSE: Evaluate Bessel function of first kind and order  
//          1 at input x                                      
// https://www.atnf.csiro.au/computing/software/gipsy/sub/bessel.c
//------------------------------------------------------------
{
   double ax,z;
   double xx,y,ans,ans1,ans2;

   if ((ax=fabs(x)) < 8.0) {
      y=x*x;
      ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
         +y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
      ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
         +y*(99447.43394+y*(376.9991397+y*1.0))));
      ans=ans1/ans2;
   } else {
      z=8.0/ax;
      y=z*z;
      xx=ax-2.356194491;
      ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
         +y*(0.2457520174e-5+y*(-0.240337019e-6))));
      ans2=0.04687499995+y*(-0.2002690873e-3
         +y*(0.8449199096e-5+y*(-0.88228987e-6
         +y*0.105787412e-6)));
      ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
      if (x < 0.0) ans = -ans;
   }
   return ans;
}

double bessj( int n, double x )
//------------------------------------------------------------
// PURPOSE: Evaluate Bessel function of first kind and order  
//          n at input x                                      
// The function can also be called for n = 0 and n = 1.   
// https://www.atnf.csiro.au/computing/software/gipsy/sub/bessel.c    
//------------------------------------------------------------
{
   int    j, jsum, m;
   double ax, bj, bjm, bjp, sum, tox, ans;


   if (n < 0)
   {
      double   dblank = 0;              // mod
      //setdblank_c( &dblank );         // mod
      return( dblank );
   }
   ax=fabs(x);
   if (n == 0)
      return( bessj0(ax) );
   if (n == 1)
      return( bessj1(ax) );
      

   if (ax == 0.0)
      return 0.0;
   else if (ax > (double) n) {
      tox=2.0/ax;
      bjm=bessj0(ax);
      bj=bessj1(ax);
      for (j=1;j<n;j++) {
         bjp=j*tox*bj-bjm;
         bjm=bj;
         bj=bjp;
      }
      ans=bj;
   } else {
      tox=2.0/ax;
      m=2*((n+(int) sqrt(ACC*n))/2);
      jsum=0;
      bjp=ans=sum=0.0;
      bj=1.0;
      for (j=m;j>0;j--) {
         bjm=j*tox*bj-bjp;
         bjp=bj;
         bj=bjm;
         if (fabs(bj) > BIGNO) {
            bj *= BIGNI;
            bjp *= BIGNI;
            ans *= BIGNI;
            sum *= BIGNI;
         }
         if (jsum) sum += bj;
         jsum=!jsum;
         if (j == n) ans=bjp;
      }
      sum=2.0*sum-bj;
      ans /= sum;
   }
   return  x < 0.0 && n%2 == 1 ? -ans : ans;
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

//-----------------------------------------------------------------------------
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
    //cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    //cv::resizeWindow(window_name, 2*img_w, img_h);

    // do work here
    try
    {    
        const cv::Range xgv = cv::Range(1, 3);
        const cv::Range ygv = cv::Range(0, 5);
        cv::Mat Y;
        //meshgrid(xgv, ygv, X, Y);

        //meshgrid(-31.5, 31.5, 1.0, -31.5, 31.5, 1.0, X, Y);

        //cv::Mat r;
        //cv::sqrt(cv::abs(X.mul(X)) + cv::abs(Y.mul(Y)), r);
        //cv::Mat circ = cv::Mat(64, 64, CV_32FC1, cv::Scalar::all(0.0));

        //for (idx = 0; idx < 64; ++idx)
        //{
        //    for (jdx = 0; jdx < 64; ++jdx)
        //    {
        //        if (r.at<double>(idx, jdx) < 31)
        //            circ.at<float>(idx, jdx) = 1.0;
        //    }
        //}
        
        //cv::circle(circ, cv::Point(31, 31), 31, 255, 0, cv::LineTypes::LINE_8, 0);
        
        //-----------------------------------------------------------------------------
        // test code
        cv::RNG rng(123456);

        //-----------------------------------------------------------------------------

        bp = 1;

        cv::Mat img = cv::imread("../../data/checker_board_32x32.png", cv::IMREAD_ANYCOLOR);
        if (img.channels() >= 3)
        {
            img.convertTo(img, CV_64FC3);
            img = get_channel(img, 1);
        }
        else
        {
            img.convertTo(img, CV_64FC1);
        }

        uint32_t N = 512;
        double pixel = 0.0125;
        double D = 0.095;
        double L = 1000;
        double wavelenth = 525e-9;
        double obj_size = N * pixel;
        double r0 = 0.0386;

        param_obj P(N, D, L, r0, wavelenth, obj_size);

        cv::Mat s_half;
        generate_psd(P);

        cv::Mat img_tilt;
        generate_tilt_image(img, P, rng, img_tilt);


        bp = 2;


    }
    catch(std::exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
    }


    std::cout << "End of Program.  Press Enter to close..." << std::endl;
	std::cin.ignore();
    cv::destroyAllWindows();

}   // end of main

