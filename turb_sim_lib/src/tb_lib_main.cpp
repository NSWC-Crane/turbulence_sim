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
void apply_turbulence(unsigned int img_w, unsigned int img_h, unsigned char *img_, unsigned char *turb_img_)
{
    
    cv::Mat img_tilt;
    
    // convert the image from pointer to cv::Mat
    cv::Mat img = cv::Mat(img_h, img_w, CV_8UC1, img_);
    cv::Mat img_blur = cv::Mat(img_h, img_w, CV_8UC1, turb_img_);
    
    
    generate_tilt_image(img, tp, rng, img_tilt);

    generate_blur_image(img_tilt, tp, rng, img_blur);
    
    
}   // end of apply_turbulence


