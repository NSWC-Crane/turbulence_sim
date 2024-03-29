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
//#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/video.hpp>
//#include <opencv2/imgcodecs.hpp>


// custom includes
#include <num2string.h>
#include <file_ops.h>
#include <opencv_helper.h>
#include <lens_pixel_size.h>

#define USE_LIB

#if defined(USE_LIB)
//#include "turb_sim_lib.h"

//typedef void (*lib_init_turbulence_generator)(unsigned int N_, double D_, double L_, double Cn2_, double obj_size_, char uc_);
typedef void (*lib_init_turbulence_generator)(char uc_);
typedef void (*lib_add_turbulence_param)(unsigned int N_, double D_, double L_, double Cn2_, double obj_size_);
typedef void (*lib_apply_turbulence)(unsigned int tp_index, unsigned int img_w, unsigned int img_h, double* img_, double* turb_img_);
typedef void (*lib_apply_rgb_turbulence)(unsigned int tp_index, unsigned int img_w, unsigned int img_h, double* img_, double* turb_img_);
typedef void (*lib_get_rgb_psf)(unsigned int tp_index, unsigned int* img_w, unsigned int* img_h, double* psf_t);
typedef void (*lib_apply_tilt)(unsigned int tp_index, unsigned int img_w, unsigned int img_h, double* img_, double* tilt_img_);

#else
#include "turbulence_param.h"
#include "turbulence_sim.h"

#endif

//-----------------------------------------------------------------------------
void create_gaussian_kernel(int32_t size, double sigma, cv::Mat& kernel)
{
    // assumes a 0 mean Gaussian distribution
    int32_t row, col;
    double s = 2 * sigma * sigma;

    kernel = cv::Mat::zeros(size, size, CV_64FC1);

    double t = (1.0 / (CV_PI * s));

    for (row = 0; row < size; ++row)
    {
        for (col = 0; col < size; ++col)
        {
            kernel.at<double>(row, col) = t * std::exp((-((col - (size >> 1)) * (col - (size >> 1))) - ((row - (size >> 1)) * (row - (size >> 1)))) / (s));
        }
    }

    double matsum = (double)cv::sum(kernel)[0];

    kernel = kernel * (1.0 / matsum);	// get the matrix to sum up to 1...

}	// end of create_gaussian_kernel

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

    std::string window_name = "image";

    std::string lib_filename;

    std::vector<int32_t> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(2);

    std::string base_directory;
    std::string baseline_filename;
    std::string real_filename;

    cv::Mat kernel;
    create_gaussian_kernel(5, 5, kernel);

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
    cv::namedWindow("color", cv::WINDOW_NORMAL);

    // do work here
    try
    {    

        //cv::Mat r;
        //cv::sqrt(cv::abs(X.mul(X)) + cv::abs(Y.mul(Y)), r);
        cv::Mat circ = cv::Mat::zeros(80, 80, CV_64FC1);

        //for (idx = 0; idx < 64; ++idx)
        //{
        //    for (jdx = 0; jdx < 64; ++jdx)
        //    {
        //        if (r.at<double>(idx, jdx) < 31)
        //            circ.at<float>(idx, jdx) = 1.0;
        //    }
        //}
        
        //cv::circle(circ, cv::Point(39, 39), (80-2)>>1, 255, 0, cv::LineTypes::LINE_8, 0);

#if defined(USE_LIB)
    // load in the library
    #if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)

    #if defined(_DEBUG)
        lib_filename = "../../turb_sim_lib/build/Debug/turb_simd.dll";
    #else
        //lib_filename = "../../turb_sim_lib/build/Release/turb_sim.dll";
        lib_filename = "C:/Projects/turbulence_sim/turb_sim_lib/build/Release/turb_sim.dll";
    #endif

        HINSTANCE turb_lib = LoadLibrary(lib_filename.c_str());

        if (turb_lib == NULL)
        {
            throw std::runtime_error("error loading library");
        }

        lib_init_turbulence_generator init_turbulence_generator = (lib_init_turbulence_generator)GetProcAddress(turb_lib, "init_turbulence_generator");
        lib_add_turbulence_param add_turbulence_param = (lib_add_turbulence_param)GetProcAddress(turb_lib, "add_turbulence_param");
        lib_apply_turbulence apply_turbulence = (lib_apply_turbulence)GetProcAddress(turb_lib, "apply_turbulence");
        lib_apply_rgb_turbulence apply_rgb_turbulence = (lib_apply_rgb_turbulence)GetProcAddress(turb_lib, "apply_rgb_turbulence");
        lib_get_rgb_psf get_rgb_psf = (lib_get_rgb_psf)GetProcAddress(turb_lib, "get_rgb_psf");
        lib_apply_tilt apply_tilt = (lib_apply_tilt)GetProcAddress(turb_lib, "apply_tilt");

    #else
        lib_filename = "../../turb_sim_lib/build/turb_sim.so";
        void* turb_lib = dlopen(lib_filename.c_str(), RTLD_NOW);

        if (turb_lib == NULL)
        {
            throw std::runtime_error("error loading library");
        }

        lib_init_turbulence_generator init_turbulence_generator = (lib_init_turbulence_generator)dlsym(turb_lib, "init_turbulence_generator");
        lib_apply_turbulence apply_turbulence = (lib_apply_turbulence)dlsym(turb_lib, "apply_turbulence");
        lib_apply_rgb_turbulence apply_rgb_turbulence = (lib_apply_rgb_turbulence)dlsym(turb_lib, "apply_rgb_turbulence");
        lib_get_rgb_psf get_rgb_psf = (lib_get_rgb_psf)dlsym(turb_lib, "get_rgb_psf");

    #endif

#endif      
        
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
        //base_directory = "D:/data/turbulence/";
        base_directory = "C:/Projects/data/turbulence/";
        
        baseline_filename = base_directory + "ModifiedBaselines/Mod_baseline_z5000_r0700.png";
        real_filename = base_directory + "sharpest/z5000/0700/image_z05000_f48095_e07485_i00.png";


        //baseline_filename = "C:/Projects/data/dfd/ATT_20230221/processed/z5500/baseline_r900_z5500.png";
        //real_filename = "C:/Projects/data/dfd/ATT_20230221/processed/z5500/baseline_r900_z5500.png";

        //baseline_filename = "../../data/random_image_512x512.png";
        //real_filename = "../../data/random_image_512x512.png";
        
#else
        base_directory = "../../data/";
        //baseline_filename = base_directory + "checker_board_32x32.png";
        //real_filename = base_directory + "checker_board_32x32.png";

        baseline_filename = "../../data/test_image_fp1.png";
        real_filename = "../../data/test_image_fp2.png";
#endif

        bool use_color = false;

        cv::Mat img, rw_img, tmp_img, img_blur;

        if (use_color)
        {
            rw_img = cv::imread(real_filename, cv::IMREAD_ANYCOLOR);
            tmp_img = cv::imread(baseline_filename, cv::IMREAD_ANYCOLOR);   // IMREAD_ANYCOLOR, IMREAD_GRAYSCALE

            if (rw_img.channels() >= 3)
            {
                rw_img.convertTo(rw_img, CV_64FC3);
            }
            else
            {
                cv::cvtColor(rw_img, rw_img, cv::COLOR_GRAY2BGR);
                rw_img.convertTo(rw_img, CV_64FC3);
            }

            if (tmp_img.channels() >= 3)
            {
                tmp_img.convertTo(tmp_img, CV_64FC3);
                //tmp_img = get_channel(tmp_img, 1);
            }
            else
            {
                cv::cvtColor(tmp_img, tmp_img, cv::COLOR_GRAY2BGR);
                tmp_img.convertTo(tmp_img, CV_64FC3);
            }
        }
        else
        {
            rw_img = cv::imread(real_filename, cv::IMREAD_GRAYSCALE);
            tmp_img = cv::imread(baseline_filename, cv::IMREAD_GRAYSCALE);   // IMREAD_ANYCOLOR, IMREAD_GRAYSCALE

            rw_img.convertTo(rw_img, CV_64FC1);
            tmp_img.convertTo(tmp_img, CV_64FC1);
        }

        uint32_t N = tmp_img.rows;
        //img = tmp_img.clone();
        //uint32_t N = 256;
        img = tmp_img(cv::Rect(0, 0, N, N)).clone();
        rw_img = rw_img(cv::Rect(0, 0, N, N)).clone();

        cv::Mat noise = cv::Mat(rw_img.rows, rw_img.cols, CV_64FC3);
        rng.fill(noise, cv::RNG::NORMAL, 0.0, 4.0);

        //rw_img += noise;

        double D = 0.095;
        uint32_t zoom = 5000;

        double L = 700;
        //double wavelenth = 525e-9;
        double pixel = get_pixel_size(zoom, L);   // 0.004217; 0.00246

        double obj_size = N * pixel;
        double Cn2 = 1e-13;

        // cn = 1e-15 -> r0 = 0.1535, Cn = 1e-14 -> r0 = 0.0386, Cn = 1e-13 -> r0 = 0.0097
        //double r0 = 0.0097;
        //double r0 = std::exp(-0.6 * std::log(0.158625 * k * k * Cn2 * L));


#if defined(USE_LIB)
        
        if(use_color)
            img_blur = cv::Mat::zeros(N, N, CV_64FC3);
        else
            img_blur = cv::Mat::zeros(N, N, CV_64FC1);

        init_turbulence_generator(0);
        add_turbulence_param(N, D, L, Cn2, obj_size);

#else
        std::cout << "Initializing the turbulence parameters" << std::endl;
        std::vector<turbulence_param> Pv;
        L = 600;
        pixel = get_pixel_size(zoom, L);
        obj_size = N * pixel;

        //Pv.push_back(turbulence_param(N, D, L, Cn2, 639e-9, obj_size));
        //Pv.push_back(turbulence_param(N, D, L, Cn2, 525e-9, obj_size));
        //Pv.push_back(turbulence_param(N, D, L, Cn2, 471e-9, obj_size));

        Pv.push_back(turbulence_param(N, D, L, Cn2, obj_size, use_color));

        //for (idx = 0; idx < 23; ++idx)
        //{
        //    L = 10.0 * idx + 600.0;
        //    pixel = turbulence_param::get_pixel_size(zoom, L);
        //    obj_size = N * pixel;
        //    Pv.push_back(turbulence_param(N, D, L, Cn2, wavelenth, obj_size));
        //}

#endif

        //-----------------------------------------------------------------------------
        cv::Mat img_tilt;
        cv::Mat img_tilt2, img_blur2;

        //cv::Mat img_blur_r, img_blur_g, img_blur_b;
        //std::vector<cv::Mat> img_v(3);
        //std::vector<cv::Mat> img_blur_v(3);
        //std::vector<cv::Mat> img_tilt_v(3);

        cv::Mat montage, montage2;
        char key = 0;

        cv::resizeWindow(window_name, 6*N, 2*N);

        auto rng_seed = time(NULL);

        bp = 1;
        
        std::cout << "Running the turbulence generation" << std::endl;

        while(key != 'q')
        {
            start_time = std::chrono::system_clock::now();


            for (int jdx = 0; jdx < 20; ++jdx)
            {
                //rng_seed = 1672270304;// time(NULL);
                //// red - 2, green - 1, blue - 0
                //rng = cv::RNG(rng_seed);

#if defined(USE_LIB)
                //uint32_t psf_w = 0, psf_h = 0;
                //std::vector<double> psf_t(512 * 512, 0);
                //get_rgb_psf(0, &psf_w, &psf_h, psf_t.data());
                //cv::Mat psf = cv::Mat(psf_h, psf_w, CV_64FC3, psf_t.data());

                //cv::filter2D(img, img_blur, -1, psf, cv::Point(-1, -1), 0.0, cv::BorderTypes::BORDER_REPLICATE);


                //apply_turbulence(N, N, img.ptr<double>(0), img_blur.ptr<double>(0));
                //apply_rgb_turbulence(0, N, N, img.ptr<double>(0), img_blur.ptr<double>(0));

#else
                generate_tilt_image(img, Pv[0], rng, img_tilt);
                
                //rng = cv::RNG(rng_seed);
                generate_blur_rgb_image(img_tilt, Pv[0], rng, img_blur);
#endif
                //cv::imwrite("test_image_fp1_i" + num2str(jdx,"%02d") + ".png", img_blur, compression_params);

#if defined(USE_LIB)

                apply_turbulence(0, N, N, rw_img.ptr<double>(0), img_blur.ptr<double>(0));
                //apply_rgb_turbulence(0, N, N, rw_img.ptr<double>(0), img_blur.ptr<double>(0));


                //cv::filter2D(img_blur, img_blur, img_blur.type(), kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT_101);

                rng.fill(noise, cv::RNG::NORMAL, 0.0, 3.0);
                //img_blur += noise;

#else
                generate_tilt_image(rw_img, Pv[0], rng, img_tilt);

                //rng = cv::RNG(rng_seed);
                generate_blur_rgb_image(img_tilt, Pv[0], rng, img_blur);
#endif
                //cv::imwrite("test_image_fp2_i" + num2str(jdx, "%02d") + ".png", img_blur, compression_params);
                //cv::imwrite("test_image_i" + num2str(jdx, "%02d") + ".png", img_blur, compression_params);

            }


            stop_time = std::chrono::system_clock::now();
            elapsed_time = std::chrono::duration_cast<d_sec>(stop_time - start_time);

            std::cout << "time (s): " << elapsed_time.count() << std::endl;

            //cv::hconcat(img_blur_v[0], img_blur_v[1], montage);
            //cv::hconcat(img, img_tilt, montage);
            cv::hconcat(rw_img, img_blur, montage2);

            //cv::imshow(window_name, montage / 255.0);
            cv::imshow("color", montage2 / 255.0);

            key = 'q';// cv::waitKey(1);
        }
        bp = 2;

    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }

    //cv::destroyAllWindows();
    std::cout << std::endl << "End of Program.  Press Enter to close..." << std::endl;
	std::cin.ignore();

}   // end of main

