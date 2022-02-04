
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

// MZA includes 
#include "MzaShare.h"
#include "MzaSysEng.h"
#include "MzaSysEngSupport.h"
#include "ParamStruct.h"
#include "MzaSciCompSupport.h"
#include "H5ReadWrite.h"

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgcodecs.hpp>

// custom includes
#include <num2string.h>
#include <file_ops.h>
#include <file_parser.h>


cv::Mat vectorToMat(double* vec, int nRows, int nCols)
{
    cv::Mat mat(nRows, nCols, CV_64FC1);

    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {
            mat.at<double>(i, j) = *vec++;
        }
    }
    return mat;
}


double spherical_r0(double lambda, std::vector<double> Cn2, double L, bool ind = true)
{
    const double k = 2 * CV_PI / lambda;
    const double b0 = 0.158625;
    double prod_terms = b0 * k * k * Cn2.at(0) * L;
    return std::pow(prod_terms, (-3.0 / 5.0));
}


// Basic example computes time-averaged blurring and applies to an image
int BasicBlurringExample(char** errorChain, cv::Mat img, cv::Mat& imgBlur,
    std::vector<double> Cn2, double L)
{
    int errorCode = 0;
    int nX, nY;
    std::vector<double> x, y;
    double lambda = 1e-6, TxD = 0.5, hp = 100, ht = 10, rd = 5000, re = 0, L;

    nX = img.rows;
    nY = img.cols;

    generate_range(-0.64, 0.6375, 0.0025, x);
    generate_range(-0.64, 0.6375, 0.0025, y);

    // Setup geometry, screens
    // double RP[3], RT[3], VP[3], VT[3], K[3];
    // returns position and velocity vectors for Platform and Target: RP, RT, VP, VT 
    // SimpleGeom(errorChain, &hp, 1, &ht, 1, &rd, 1, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
    //    RP, RT, VP, VT, &re);

    // DifferenceElementwise(errorChain, 3, RT, RP, K);
    // vector2Norm(errorChain, 3, K, 1, &L);
    // const int nscreens = 100;

    // double screenX[nscreens], screenDx[nscreens], z[nscreens], dz[nscreens], h[nscreens];
    // double xRange[2] = { 0, 1 };

    // compute the phase screens distance (z) and thickness (dz) 
    // computeScreensECF(errorChain,
        // inputs: nScreens, x0, dx0, maxAlt, xRange0, nGeoms, posPlat, posTarg, earthRadius, nEarthRadii,
        // nscreens, NULL, NULL, NULL, NULL, 1, RP, RT, &re, 1,
        // outputs: x, dx, z, dz, h, xRange
        // screenX, screenDx, z, dz, h, xRange);

    // double Cn2[nscreens] = {};
    // HV57(errorChain, h, nscreens, Cn2);
    // double r0 = 0;
    // SphericalR0(errorChain, &lambda, Cn2, &L, z, dz, 1, 100, &r0);
    // std::vector<double> cn2_vec(std::begin(Cn2), std::end(Cn2));
    //double r01 = spherical_r0(lambda, cn2_vec, L);
    double r0 = spherical_r0(lambda, Cn2, L);

    // Compute higher-order OTF
    int M = (nX > nY) ? getNextPower2(nX) : getNextPower2(nY);
    std::vector<double> fx(M);
    std::vector<double> fy(M);
    std::vector<double> OTFHO(M * M);
    double Sh[] = { 1.0 };    // specify a Strehl value, use Sh = 1 -> diffraction-only
    //double* Sh = NULL; // use open-loop Strehl value
    errorCode = HigherOrderBlurSpec(errorChain, nY, y.data(), nX, x.data(), 'F', lambda, TxD, L, r0, Sh, M,
        OTFHO.data(), fx.data(), fy.data());
    if (errorCode < 0) { return errorCheck(errorChain, "BasicBlurringExample", SC_ERRSTATUS, "when calling HigherOrderBlurSpec."); }

    // Compute OTF for time-averaged jitter
    std::vector<double> OTFTilt(M * M);
    double sigP = 0;
    double sigT = 0; // jitter values (rad)
    //sigP = 10.0e-6, sigT = 0e-6; // specify jitter values
    OpenLoopJitter(errorChain, &r0, &TxD, &lambda, 1, &sigP); sigT = sigP; // compute open-loop jitter values
    errorCode = JitterBlurSpec(errorChain, nY, y.data(), nX, x.data(), L, sigP, sigT, M,
        OTFTilt.data(), fx.data(), fy.data());
    if (errorCode < 0) { return errorCheck(errorChain, "BasicBlurringExample", SC_ERRSTATUS, "when calling JitterBlurSpec."); }

    std::vector<double> BlurSpec(M * M);
    ProductElementwise(errorChain, M * M, OTFHO.data(), OTFTilt.data(), BlurSpec.data()); // combined OTF

    // Apply combined OTF to image
    double dx0;
    std::vector<double> diffx(nX - 1);
    Difference(errorChain, x.data(), nX, diffx.data());
    Maximum(errorChain, diffx.data(), nX - 1, &dx0);
    double dy0;
    std::vector<double> diffy(nY - 1);
    Difference(errorChain, y.data(), nY, diffy.data());
    Maximum(errorChain, diffy.data(), nY - 1, &dy0);
    double dA = sqrt(dx0 * dy0);

    std::vector<double> imgOut(nX * nY);
    errorCode = ApplyBlurSpec(errorChain, nX, nY, img.ptr<double>(0), M, M, BlurSpec.data(), &dA, imgOut.data());
    if (errorCode < 0) { return errorCheck(errorChain, "BasicBlurringExample", SC_ERRSTATUS, "when calling ApplyBlurSpec."); }

    imgBlur = vectorToMat(imgOut.data(), nX, nY);
    cv::multiply(255, imgBlur, imgBlur);
    imgBlur.convertTo(imgBlur, CV_8UC1);

    return errorCode;
}


// ----------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string data_dir = "../../data/";
    std::string filename = "checker_board_32x32.png";
    cv::Mat img, imgOut;

    int errorCode;
    std::string tmp(2000, ' ');
    char* errorChain[1];
    errorChain[0] = &tmp[0];

    std::string window_name = "Blurring Example";
    std::vector<double> Cn2;
    Cn2.push_back(1e-14);


    // do work here
    try
    {
        // read in image
        img = cv::imread(data_dir + filename, cv::IMREAD_GRAYSCALE);
        img.convertTo(img, CV_64FC1);
        cv::multiply(1.0 / 255.0, img, img);

        cv::namedWindow(window_name, cv::WINDOW_NORMAL);

        for (int range = 600; range <= 1000; range = range + 50)
        {
            BasicBlurringExample(errorChain, img, imgOut, Cn2, range);
            cv::imshow(window_name, imgOut);
            cv::waitKey(0);
        }

        cv::imwrite("test_blur.png", imgOut);
    }
    catch (std::exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
    }

    cv::destroyAllWindows();

    std::cout << "End of Program.  Press Enter to close..." << std::endl;
    std::cin.ignore();

}   // end of main
