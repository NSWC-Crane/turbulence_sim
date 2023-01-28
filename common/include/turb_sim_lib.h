#ifndef _TURBULENCE_SIMULATION_LIBRARY_H_
#define _TURBULENCE_SIMULATION_LIBRARY_H_


#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)

    #ifdef BUILD_LIB
        #ifdef LIB_EXPORTS
            #define TS_LIB __declspec(dllexport)
        #else
            #define TS_LIB __declspec(dllimport)
        #endif
    #else
        #define TS_LIB
    #endif

#else
    #define TS_LIB

#endif

//-----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

    //-----------------------------------------------------------------------------
    // perform the initilization of the turbulence params
    //TS_LIB void init_turbulence_generator(unsigned int N_, double D_, double L_, double Cn2_, double obj_size_, char uc_);
    TS_LIB void init_turbulence_generator(char uc_);

    //-----------------------------------------------------------------------------
    // adds another turbulence parameter
    TS_LIB void add_turbulence_param(unsigned int N_, double D_, double L_, double Cn2_, double obj_size_);

    //-----------------------------------------------------------------------------
    // allows the user to set the random number generator seed whenever they want
    TS_LIB void set_rng_seed(size_t seed);

    //-----------------------------------------------------------------------------
    // function to update the Cn2 value for all of the turbulence parameters
    TS_LIB void update_cn2(double Cn2_);

    //-----------------------------------------------------------------------------
    // apply the turbulence to a single channel image
    TS_LIB void apply_turbulence(unsigned int tp_index, unsigned int img_w, unsigned int img_h, double *img_, double *turb_img_);

    //-----------------------------------------------------------------------------
    // apply the turbulence to a 3-color RGB image - image must be in BGR format
    TS_LIB void apply_rgb_turbulence(unsigned int tp_index, unsigned int img_w, unsigned int img_h, double* img_, double* turb_img_);

#ifdef __cplusplus
}
#endif

#endif  // _TURBULENCE_SIMULATION_LIBRARY_H_
