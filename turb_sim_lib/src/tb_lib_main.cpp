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
std::vector<turbulence_param> tp;
cv::RNG rng;
bool use_color;

//-----------------------------------------------------------------------------
void set_rng_seed(size_t seed)
{
    rng = cv::RNG(seed);
}   // end of set_rng_seed

//-----------------------------------------------------------------------------
//void init_turbulence_generator(unsigned int N_, double D_, double L_, double Cn2_, double obj_size_, char uc_)
//{
//    tp.clear();
//    use_color = uc_ == 1 ? true : false;
//    tp.push_back(turbulence_param(N_, D_, L_, Cn2_, obj_size_, use_color));
//
//    rng = cv::RNG(time(NULL));
//
//}   // end of init_turbulence_generator

//-----------------------------------------------------------------------------
void init_turbulence_generator(char uc_)
{
    tp.clear();
    use_color = uc_ == 1 ? true : false;
    //tp.push_back(turbulence_param(N_, D_, L_, Cn2_, obj_size_, use_color));

    rng = cv::RNG(time(NULL));

}   // end of init_turbulence_generator

//-----------------------------------------------------------------------------
//void init_turbulence_params(unsigned int N_, double D_, double L_, double Cn2_, double obj_size_, bool uc_)
//{
//    tp.resize(1);
//    tp[0].update_params(N_, D_, L_, Cn2_, obj_size_, uc_);
//    rng = cv::RNG(time(NULL));
//    
//}   // end of init_turbulence_params

void add_turbulence_param(unsigned int N_, double D_, double L_, double Cn2_, double obj_size_)
{
    tp.push_back(turbulence_param(N_, D_, L_, Cn2_, obj_size_, use_color));
}

//-----------------------------------------------------------------------------
void update_cn2(double Cn2_)
{
    uint32_t idx;

    for (idx = 0; idx < tp.size(); ++idx)
        tp[idx].update_Cn2(Cn2_);

}   // end of update_cn2

//-----------------------------------------------------------------------------
void apply_turbulence(unsigned int tp_index, unsigned int img_w, unsigned int img_h, double *img_, double *turb_img_)
{
    //uint32_t index = 0;

    cv::Mat img_tilt, img_blur;
    try
    {
        // convert the image from pointer to cv::Mat
        cv::Mat img = cv::Mat(img_h, img_w, CV_64FC1, img_);
        cv::Mat turb_img = cv::Mat(img_h, img_w, CV_64FC1, turb_img_);

        //std::cout << "img[0]: " << img.at<double>(0, 0) << "/" << img_[0] << std::endl;
        //std::cout << "turb_img[0]: " << turb_img.at<double>(0, 0) << "/" << turb_img_[0] << std::endl;

        generate_tilt_image(img, tp[tp_index], rng, img_tilt);

        //std::cout << "img[0]: " << img.at<double>(0, 0) << "/" << img_[0] << std::endl;
        //std::cout << "img_tilt[0]: " << img_tilt.at<double>(0, 0) << std::endl;
        //std::cout << "turb_img[0]: " << turb_img.at<double>(0, 0) << "/" << turb_img_[0] << std::endl;

        generate_blur_image(img_tilt, tp[tp_index], rng, turb_img);

        //img_blur.convertTo(turb_img, CV_8UC1);
    }
    catch (std::exception e)
    {
        std::string error_string = "Error: " + std::string(e.what()) + "\n";
        error_string += "File: " + std::string(__FILE__) + ", Function: " + std::string(__FUNCTION__) + ", Line #: " + std::to_string(__LINE__);
        std::cout << error_string << std::endl;
    }
    
}   // end of apply_turbulence

//-----------------------------------------------------------------------------
void apply_rgb_turbulence(unsigned int tp_index, unsigned int img_w, unsigned int img_h, double* img_, double* turb_img_)
{
    //uint32_t index = 0;
    cv::Mat img_tilt, img_blur;

    try
    {
        // convert the image from pointer to cv::Mat
        cv::Mat img = cv::Mat(img_h, img_w, CV_64FC3, img_);
        cv::Mat turb_img = cv::Mat(img_h, img_w, CV_64FC3, turb_img_);

        //std::cout << "img[0]: " << img.at<double>(0, 0) << "/" << img_[0] << std::endl;
        //std::cout << "turb_img[0]: " << turb_img.at<double>(0, 0) << "/" << turb_img_[0] << std::endl;

        generate_tilt_image(img, tp[tp_index], rng, img_tilt);

        //std::cout << "img[0]: " << img.at<double>(0, 0) << "/" << img_[0] << std::endl;
        //std::cout << "img_tilt[0]: " << img_tilt.at<double>(0, 0) << std::endl;
        //std::cout << "turb_img[0]: " << turb_img.at<double>(0, 0) << "/" << turb_img_[0] << std::endl;

        generate_blur_rgb_image(img_tilt, tp[tp_index], rng, turb_img);

        //img_blur.convertTo(turb_img, CV_8UC1);
    }
    catch (std::exception e)
    {
        std::string error_string = "Error: " + std::string(e.what()) + "\n";
        error_string += "File: " + std::string(__FILE__) + ", Function: " + std::string(__FUNCTION__) + ", Line #: " + std::to_string(__LINE__);
        std::cout << error_string << std::endl;
    }

}   // end of apply_rgb_turbulence
