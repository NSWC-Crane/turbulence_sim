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

//#include "cv_dft_conv.h"
#include "turbulence_param.h"
#include "turbulence_sim.h"

#define USE_LIB


#if defined(USE_LIB)
#include "turb_sim_lib.h"

typedef void (*lib_init_turbulence_params)(unsigned int N_, double D_, double L_, double Cn2_, double w_, double obj_size_);
typedef void (*lib_apply_turbulence)(unsigned int img_w, unsigned int img_h, double* img_, double* turb_img_);

#endif

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

    std::string window_name = "image";

    std::string lib_filename;


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
    //cv::resizeWindow(window_name, 2*img_w, img_h);

    // do work here
    try
    {    

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


#if defined(USE_LIB)
    // load in the library
    #if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
        lib_filename = "../../turb_sim_lib/build/Release/turb_sim.dll";
        HINSTANCE turb_lib = LoadLibrary(lib_filename.c_str());

        if (turb_lib == NULL)
        {
            throw std::runtime_error("error loading library");
        }

        lib_init_turbulence_params init_turbulence_params = (lib_init_turbulence_params)GetProcAddress(turb_lib, "init_turbulence_params");
        lib_apply_turbulence apply_turbulence = (lib_apply_turbulence)GetProcAddress(turb_lib, "apply_turbulence");

    #else
        lib_filename = "../../turb_sim_lib/build/turb_sim.so";
        void* turb_lib = dlopen(lib_filename.c_str(), RTLD_NOW);

        if (turb_lib == NULL)
        {
            throw std::runtime_error("error loading library");
        }

        lib_init_turbulence_params init_turbulence_params = (lib_init_turbulence_params)dlsym(turb_lib, "init_turbulence_params");
        lib_apply_turbulence apply_turbulence = (lib_apply_turbulence)dlsym(turb_lib, "apply_turbulence");

    #endif

#endif      

        bp = 1;
//        std::string filename = "../../data/checker_board_32x32.png";
        //std::string filename = "D:/data/turbulence/sharpest/z5000/baseline_z5000_r1000.png";
        std::string filename = "C:/Projects/data/turbulence/sharpest/z5000/baseline_z5000_r1000.png";
        cv::Mat img;
        cv::Mat tmp_img = cv::imread(filename, cv::IMREAD_ANYCOLOR);

        if (tmp_img.channels() >= 3)
        {
            //tmp_img.convertTo(tmp_img, CV_64FC3, 1.0 / 255.0);
            tmp_img.convertTo(tmp_img, CV_64FC3);
            tmp_img = get_channel(tmp_img, 1);
        }
        else
        {
            //tmp_img.convertTo(tmp_img, CV_64FC1, 1.0 / 255.0);
            tmp_img.convertTo(tmp_img, CV_64FC1);
        }

        //uint32_t N = tmp_img.rows;
        //img = tmp_img.clone();
        uint32_t N = 200;
        img = tmp_img(cv::Rect(0, 0, N, N));

        double pixel = 0.004217;    // 0.004217; 0.00246
        double D = 0.095;
        double L = 1000;
        double wavelenth = 525e-9;
        double obj_size = N * pixel;
        //double k = 2 * CV_PI / wavelenth;
        double Cn2 = 1e-13;
        // cn = 1e-15 -> r0 = 0.1535, Cn = 1e-14 -> r0 = 0.0386, Cn = 1e-13 -> r0 = 0.0097
        //double r0 = 0.0097;
        //double r0 = std::exp(-0.6 * std::log(0.158625 * k * k * Cn2 * L));

#if defined(USE_LIB)
        init_turbulence_params(N, D, L, Cn2, wavelenth, obj_size);
#else
        turbulence_param P(N, D, L, Cn2, wavelenth, obj_size);
#endif

        //-----------------------------------------------------------------------------
        cv::Mat img_tilt;
        cv::Mat img_blur = cv::Mat::zeros(N, N, CV_64FC1);
        cv::Mat montage;
        char key = 0;

        cv::resizeWindow(window_name, 4*N, 2*N);

        while(key != 'q')
        {
            start_time = std::chrono::system_clock::now();



#if defined(USE_LIB)

            apply_turbulence(N, N, img.ptr<double>(0), img_blur.ptr<double>(0));
#else
            generate_tilt_image(img, P, rng, img_tilt);

            generate_blur_image(img_tilt, P, rng, img_blur);
#endif

            //img_blur.convertTo(img_blur, CV_8UC1);

            stop_time = std::chrono::system_clock::now();
            elapsed_time = std::chrono::duration_cast<d_sec>(stop_time - start_time);

            std::cout << "time (s): " << elapsed_time.count() << std::endl;

            cv::hconcat(img, img_blur, montage);
            cv::imshow(window_name, montage/255.0);
            key = cv::waitKey(0);
        }
        bp = 2;

    }
    catch(std::exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
        std::cout << "Filename: " << __FILE__ << std::endl;
        std::cout << "Line #: " << __LINE__ << std::endl;
    }

    cv::destroyAllWindows();
    std::cout << "End of Program.  Press Enter to close..." << std::endl;
	std::cin.ignore();

}   // end of main

