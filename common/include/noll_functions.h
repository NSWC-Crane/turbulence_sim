#ifndef _NOLL_FUNCTIONS_H_
#define _NOLL_FUNCTIONS_H_

#include <cstdint>
#include <cmath>
#include <vector>
#include <iostream>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Custom includes


//-----------------------------------------------------------------------------
//This function maps the input "j" to the (row, column) of the Zernike pyramid using the Noll numbering scheme.
//
//Authors: Tim van Werkhoven, Jason Saredy
//See: https://github.com/tvwerkhoven/libtim-py/blob/master/libtim/zern.py
//
void noll2zernike_index(int64_t j, int64_t &n, int64_t &m)
{

    if (j == 0)
        std::cout << "Noll indices start at 1, 0 is invalid." << std::endl;
        
    n = 0;
    int64_t j1 = j - 1;
    
    while (j1 > n)
    {
        ++n;
        j1 -= n;
    }
    
    //m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))
    m = (((j & 0x01) == 1) ? -1 : 1) * ((n % 2) + 2 * ((j1 + ((n + 1) % 2))>>1));

}   // end of noll_2_zern_index


//-----------------------------------------------------------------------------
//This function generates the covariance matrix for a single point source. See the associated paper for details on
//the matrix itself.

//:param Z: Number of Zernike basis functions/coefficients, determines the size of the matrix.
//:param D: The diameter of the aperture (meters)
//:param fried: The Fried parameter value
//
cv::Mat noll_covariance_matrix(uint32_t Z, double D, double r0)
{   
    uint32_t idx, jdx;
    double c1, num, den;

    int64_t ni, mi, nj, mj;

    cv::Mat dst = cv::Mat::zeros(Z, Z, CV_64FC1);

    for (idx=0; idx<Z; ++idx)
    {
        for (jdx=0; jdx<Z; ++jdx)
        {
            //ni, mi = nollToZernInd(idx+1)
            //nj, mj = nollToZernInd(jdx+1)
            noll2zernike_index(idx + 1, ni, mi);
            noll2zernike_index(jdx + 1, nj, mj);
                       
            // if (abs(mi) == abs(mj)) and (np.mod(i - j, 2) == 0):
            if ((abs(mi) == abs(mj)) & ((idx - jdx) % 2 == 0))
            {
                // num = math.gamma(14.0/3.0) * math.gamma((ni + nj - 5.0/3.0)/2.0)
                num = std::tgamma(14.0/3.0) * std::tgamma(((double)(ni + nj) - 5.0/3.0)/2.0);
                
                // den = math.gamma((-ni + nj + 17.0/3.0)/2.0) * math.gamma((ni - nj + 17.0/3.0)/2.0) * math.gamma((ni + nj + 23.0/3.0)/2.0)
                den = std::tgamma(((double)(-ni + nj) + 17.0/3.0)/2.0) * std::tgamma(((double)(ni - nj) + 17.0/3.0)/2.0) * std::tgamma(((double)(ni + nj) + 23.0/3.0)/2.0);
                      
                // coef1 = 0.0072 * (np.pi ** (8.0/3.0)) * ((D/fried) ** (5.0/3.0)) * np.sqrt((ni + 1) * (nj + 1)) * ((-1) ** ((ni + nj - 2*abs(mi))/2.0))
                // x^(y) = std::exp(y * std::log(x))                
                c1 = 0.0072 * std::exp((8.0 / 3.0) * std::log(CV_PI)) * std::exp((5.0 / 3.0) * std::log(D / r0));
                c1 *= std::sqrt((ni + 1) * (nj + 1)) * (((((ni + nj - 2 * std::abs(mi)) >> 1) & 0x01) == 1) ? -1.0 : 1.0);

                //C[i, j] = coef1*num/den
                dst.at<double>(idx, jdx) = c1*num/den;
            }
            else
            {
                dst.at<double>(idx, jdx) = 0.0;
            }
        }
    }
    
    dst.at<double>(0,0) = 1.0;

    return dst;
}   // end of noll_covariance_matrix


#endif  // _NOLL_FUNCTIONS_H_