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
    TS_LIB void init_turbulence_params(unsigned int N_, double D_, double L_, double Cn2_, double w_, double obj_size_);

    //-----------------------------------------------------------------------------
    // apply the turbulence to the image
    TS_LIB void apply_turbulence(unsigned int img_w, unsigned int img_h, double *img_, double *turb_img_);

#ifdef __cplusplus
}
#endif

#endif  // _TURBULENCE_SIMULATION_LIBRARY_H_