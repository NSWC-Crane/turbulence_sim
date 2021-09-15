#ifndef _ZERNIKE_FUNCTIONS_H_
#define _ZERNIKE_FUNCTIONS_H_

#include <cstdint>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//-----------------------------------------------------------------------------
void noll_2_zern_index(int32_t j, int32_t &n, int32_t &m)
{
    /*
    This function maps the input "j" to the (row, column) of the Zernike pyramid using the Noll numbering scheme.

    Authors: Tim van Werkhoven, Jason Saredy
    See: https://github.com/tvwerkhoven/libtim-py/blob/master/libtim/zern.py
    */
    
    if (j == 0)
        std::cout << "Noll indices start at 1, 0 is invalid." << std::endl;
        
    n = 0;
    int32_t j1 = j - 1;
    
    while (j1 > n)
    {
        ++n;
        j1 -= n;
    }
    
    //m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))
    m = (((j & 0x01) == 1) ? -1 : 1) * ((n % 2) + 2 * ((j1 + ((n + 1) % 2))>>1));

}   // end of noll_2_zern_index

//-----------------------------------------------------------------------------
cv::Mat noll_cov_matrix(Z, D, fried):
{
    /*
    This function generates the covariance matrix for a single point source. See the associated paper for details on
    the matrix itself.

    :param Z: Number of Zernike basis functions/coefficients, determines the size of the matrix.
    :param D: The diameter of the aperture (meters)
    :param fried: The Fried parameter value
    :return:
    */
    
    int32_t idx, jdx;
    float coef1, num, den;
    cv::Mat cov = cv::Mat::zeros(Z, Z, CV_64FC1);
    //C = np.zeros((Z,Z))
    
    
    for (idx=0; idx<Z; ++idx)
    {
        for (jdx=0; jdx<Z; ++jdx)
        {
            //ni, mi = nollToZernInd(idx+1)
            //nj, mj = nollToZernInd(jdx+1)
            
            noll_2_zern_index(idx+1, ni, mi);
            noll_2_zern_index(jdx+1, nj, mj);
                       
            if (abs(mi) == abs(mj)) and (np.mod(idx - jdx, 2) == 0):
            {
                num = tgamma(14.0/3.0) * tgamma(((float)(ni + nj) - 5.0/3.0)/2.0);
                
                den = tgamma(((float)(-ni + nj) + 17.0/3.0)/2.0) * tgamma(((float)(ni - nj) + 17.0/3.0)/2.0) * tgamma(((float)(ni + nj) + 23.0/3.0)/2.0);
                      
                coef1 = 0.0072 * powf(cv::pi,(8.0/3.0)) * powf((D/fried), (5.0/3.0)) * sqrt((ni + 1) * (nj + 1)) * ((((ni + nj - 2*abs(mi)) >> 1)&0x01 == 1) ? -1.0 : 1.0);
                        
                //C[i, j] = coef1*num/den
                cov.at<float>(idx, jdx) = coef1*num/den;
            }
            else
            {
                cov.at<float>(idx, jdx) = 0.0;
            }
        }
    }
    
    cov.at<float>(0,0) = 1.0;
    return cov;
}   // end of noll_cov_matrix
 

//-----------------------------------------------------------------------------

cv::Mat gen_zernike_coeff(num_zern, d_r0):
{
    /*
    Just a simple function to generate random coefficients as needed, conforms to Zernike's Theory. The nollCovMat()
    function is at the heart of this function.

    A note about the function call of nollCovMat in this function. The input (..., 1, 1) is done for the sake of
    flexibility. One can call the function in the typical way as is stated in its description. However, for
    generality, the D/r0 weighting is pushed to the "b" random vector, as the covariance matrix is merely scaled by
    such value.

    :param num_zern: This is the number of Zernike basis functions/coefficients used. Should be numbers that the pyramid
    rows end at. For example [1, 3, 6, 10, 15, 21, 28, 36]
    :param D_r0:
    :return:
    */
    
    // C = nollCovMat(num_zern, 1, 1)
    // e_val, e_vec = np.linalg.eig(C)
    // R = np.real(e_vec * np.sqrt(e_val))

    // b = np.random.randn(int(num_zern), 1) * D_r0 ** (3.0/10.0)
    // a = np.matmul(R, b)

    // return a
}





#endif  // _ZERNIKE_FUNCTIONS_H_