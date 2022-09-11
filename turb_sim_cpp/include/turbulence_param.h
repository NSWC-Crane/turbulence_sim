#ifndef _TURBULENCE_PARAMETERS_H_
#define _TURBULENCE_PARAMETERS_H_

#include <cstdint>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class param_obj
{
public:
    std::vector<std::complex<double>> S_vec;

    param_obj(uint32_t N_, double D_, double L_, double r0_, double w_, double obj_size_) : N(N_), D(D_), L(L_), r0(r0_), wavelength(w_), obj_size(obj_size_)
    {
        init_params();
    }
    
    //-----------------------------------------------------------------------------
    void init_params(void)
    {
        D_r0 = D / r0;
        delta0 = (L * wavelength) / (2.0 * D);
        k = (2.0 * CV_PI) / wavelength;
        s_max = (delta0 / D) * (double)N;   //???????
        spacing = delta0 / D;
        ob_s = obj_size;
        scaling = obj_size / (N * delta0);

        s_max *= scaling;
        spacing *= scaling;
        ob_s *= scaling;
        delta0 *= scaling;
    }
    
    //-----------------------------------------------------------------------------
    void update_params(uint32_t N_, double D_, double L_, double r0_, double w_, double obj_size_)
    {
        N = N_;
        D = D_;
        L = L_;
        r0 = r0_;
        wavelength = w_;
        obj_size = obj_size_;

        init_params();
        
    }

    //-----------------------------------------------------------------------------
    void update_params(double L_)
    {
        L = L_;
        init_params();
    }

    //-----------------------------------------------------------------------------
    void update_params(double L_, double D_)
    {
        D = D_;
        L = L_;

        init_params();
        
    }
    

    //-----------------------------------------------------------------------------
    double get_D(void) { return D; }
    void set_D(double D_) 
    { 
        D = D_; 
        init_params();
    }

    //-----------------------------------------------------------------------------
    double get_L(void) { return L; }
    void set_L(double L_) 
    { 
        L = L_; 
        init_params();
    }
   
    //-----------------------------------------------------------------------------
    uint64_t get_N(void) { return N; }
    void set_N(uint64_t N_) 
    { 
        N = N_; 
        init_params();
    }

    //-----------------------------------------------------------------------------
    double get_wavelength(void) { return wavelength; }
    void set_wavelength(double w_) 
    { 
        wavelength = w_; 
        init_params();
    }

    //-----------------------------------------------------------------------------
    cv::Mat get_S(void) { return S; }
    void set_S(cv::Mat S_) { S = S_.clone(); }

    //-----------------------------------------------------------------------------
    std::vector<std::complex<double>> get_S_vec(void) { return S_vec; }
    void set_S_vec(std::vector<std::complex<double>> S_vec_) { S_vec = S_vec_; }

    //-----------------------------------------------------------------------------
    double get_D_r0(void) { return D_r0; }

    //-----------------------------------------------------------------------------
    double get_delta0(void) { return delta0; }

//-----------------------------------------------------------------------------
private:
    uint64_t N;             // size of pixels of one image dimension (assumed to be square image N x N)
    double D;               // size of aperture diameter (meters)
    double L;               // length of propagation (meters)
    double r0;              // Fried parameter (meters)
    double wavelength;      // wavelength (meters)
    double obj_size;
    
    double D_r0;
    double delta0;
    double k;
    double s_max;
    double spacing;
    double ob_s;
    double scaling;
    
    cv::Mat S;


};


#endif  // _TURBULENCE_PARAMETERS_H_
