#ifndef _CV_DFT_CONV_H_
#define _CV_DFT_CONV_H_

#include <cstdint>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

const double pi = 3.14159265358979;


/*
def p_obj(N, D, L, r0, wvl, obj_size):
    """
    The parameter "object". It is really just a list of some useful parameters.
    :param N: size of pixels of one image dimension (assumed to be square image N x N).
    :param D: size of aperture diameter (meters)
    :param L: length of propagation (meters)
    :param r0: Fried parameter (meters)
    :param wvl: wavelength (meters)
    :return: returns the parameter object
    """
    a = {}
    a['N'] = N
    a['D'] = D
    a['L'] = L
    a['wvl'] = wvl
    a['r0'] = r0
    a['Dr0'] = D/r0
    a['delta0'] = L*wvl/(2*D)
    a['k'] = 2*np.pi/wvl
    a['smax'] = a['delta0']/D*N
    a['spacing'] = a['delta0']/D
    a['ob_s'] = obj_size
    a['scaling'] = obj_size / (N * a['delta0'])

    a['smax'] *= a['scaling']
    a['spacing'] *= a['scaling']
    a['ob_s'] *= a['scaling']
    a['delta0'] *= a['scaling']

    return a
    */

class turbulence_params
{
public:
    uint32_t N;         /**< size of pixels of one image dimension (assumed to be square image N x N) */
    double D;           /**< size of aperture diameter (meters) */
    double L;           /**< length of propagation (meters) */
    double wvl;         /**< wavelength (meters) */
    double r0;          /**< Fried parameter (meters) */
    double obj_size;    /**< */


    turbulence_params() = default;

    turbulence_params(uint32_t N_, double D_, double L_, double wvl_, double r0_, double obj_) :
        N(N_), D(D_), L(L_), wvl(wvl_), r0(r0_), obj_size(obj_)
    {
        Dr0 = D / r0;
        delta0 = L * wvl / (2.0 * D);
        k = 2 * pi / wvl;
        smax = delta0 / D * N;
        spacing = delta0 / D;
        scaling = obj_size / (N * delta0);

        //smax *= scaling;
        spacing *= scaling;
        obj_size *= scaling;
        delta0 *= scaling;
    }



    /*
    def gen_PSD(p_obj):
    """
    This function generates the PSD necessary for the tilt values (both x and y pixel shifts). The PSD is **4 times**
    the size of the image, this is to simplify the generation of the random vector using a property of Toeplitz
    matrices. This is further highlighted in the genTiltImg() function, where only 1/4 of the entire grid is used
    (this is because of symmetry about the origin -- hence why the PSD is quadruple the size).
    All that is required is the parameter list, p_obj.
    :param p_obj:
    :return: PSD
    """
    N = 2 * p_obj['N']
    smax = p_obj['delta0'] / p_obj['D'] * N
    c1 = 2 * ((24 / 5) * gamma(6 / 5)) ** (5 / 6)
    c2 = 4 * c1 / np.pi * (gamma(11 / 6)) ** 2
    s_arr = np.linspace(0, smax, N)
    I0_arr = np.float32(s_arr * 0)
    I2_arr = np.float32(s_arr * 0)
    for i in range(len(s_arr)):
        I0_arr[i] = I0(s_arr[i])
        I2_arr[i] = I2(s_arr[i])

    i, j = np.int32(N / 2), np.int32(N / 2)

    [x, y] = np.meshgrid(np.arange(1, N + 0.01, 1), np.arange(1, N + 0.01, 1))

    s = np.sqrt((x - i) ** 2 + (y - j) ** 2)

    C = (In_m(s, p_obj['delta0'] / p_obj['D'] * N , I0_arr) + In_m(s, p_obj['delta0'] / p_obj['D'] * N, I2_arr)) / I0(0)

    C[round(N / 2), round(N / 2)] = 1
    C = C * I0(0) * c2 * (p_obj['Dr0']) ** (5 / 3) / (2 ** (5 / 3)) * (2 * p_obj['wvl'] / (np.pi * p_obj['D'])) ** 2 * 2 * np.pi
    Cfft = np.fft.fft2(C)
    S_half = np.sqrt(Cfft)
    S_half_max = np.max(np.max(np.abs(S_half)))
    S_half[np.abs(S_half) < 0.0001 * S_half_max] = 0

    return S_half
    */

    void gen_psd(cv::Mat &psd)
    {
        double c1 = 2.0 * pow(((24 / 5) * tgamma(1.20)), (1.20));
        double c2 = 4 * c1 / pi * (tgamma(11.0 / 6.0)*tgamma(11.0 / 6.0));

        //s_arr = np.linspace(0, smax, N)



    }




private:
    double Dr0;
    double delta0;
    double k;
    double smax;
    double spacing;
    double scaling;


};


#endif	// _CV_DFT_CONV_H_