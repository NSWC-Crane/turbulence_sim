#define _CRT_SECURE_NO_WARNINGS

#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
//#include <windows.h>

#else
//#include <dlfcn.h>
//typedef void* HINSTANCE;

#endif

// C/C++ includes
#include <cmath>
#include <ctime>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>


// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// custom includes
#include "turbulence_param.h"
#include "turbulence_sim.h"

#include "turb_sim_lib.h"

// ----------------------------------------------------------------------------------------
turbulence_param tp;
cv::RNG rng;

//-----------------------------------------------------------------------------
void init_turbulence_params(unsigned int N_, double D_, double L_, double Cn2_, double w_, double obj_size_)
{
    tp.update_params(N_, D_, L_, Cn2_, w_, obj_size_);
    rng = cv::RNG(time(NULL));
    
}   // end of init_turbulence_params

//-----------------------------------------------------------------------------
double get_pixel_size(unsigned int z, double r)
{
    return tp.get_pixel_size(z, r);
}

//-----------------------------------------------------------------------------
void apply_turbulence(unsigned int img_w, unsigned int img_h, double *img_, double *turb_img_)
{
    
    cv::Mat img_tilt, img_blur;
    try
    {
        // convert the image from pointer to cv::Mat
        cv::Mat img = cv::Mat(img_h, img_w, CV_64FC1, img_);
        cv::Mat turb_img = cv::Mat(img_h, img_w, CV_64FC1, turb_img_);

        //std::cout << "img[0]: " << img.at<double>(0, 0) << "/" << img_[0] << std::endl;
        //std::cout << "turb_img[0]: " << turb_img.at<double>(0, 0) << "/" << turb_img_[0] << std::endl;

        generate_tilt_image(img, tp, rng, img_tilt);

        //std::cout << "img[0]: " << img.at<double>(0, 0) << "/" << img_[0] << std::endl;
        //std::cout << "img_tilt[0]: " << img_tilt.at<double>(0, 0) << std::endl;
        //std::cout << "turb_img[0]: " << turb_img.at<double>(0, 0) << "/" << turb_img_[0] << std::endl;

        generate_blur_image(img_tilt, tp, rng, turb_img);

        //img_blur.convertTo(turb_img, CV_8UC1);
    }
    catch (std::exception e)
    {
        std::cout << "error: " << std::endl << e.what() << std::endl;
        std::cout << "Filename: " << __FILE__ << std::endl;
        std::cout << "Line #: " << __LINE__ << std::endl << std::endl;
        std::cout << "Function: " << __FUNCTION__ << std::endl << std::endl;
    }
    
}   // end of apply_turbulence


