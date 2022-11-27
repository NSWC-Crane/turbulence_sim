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
    // take a double image and scale between 0 and 1
    TS_LIB void normalize_img(unsigned int img_w, unsigned int img_h, double* img_t, double* norm_img_t);

    //-----------------------------------------------------------------------------
    // 
    TS_LIB void transform_single_image(ms_image r_img,
        ms_image t_img,
        double* fused_data64_t,
        unsigned char* fused_data8_t
    );

    //-----------------------------------------------------------------------------
    // 
    TS_LIB void transform_multi_image(uint32_t N,
        ms_image r_img,
        ms_image* t_img,
        double* fused_data64_t,
        unsigned char* fused_data8_t
    );

    //-----------------------------------------------------------------------------
    TS_LIB void transform_multi_image_rect(uint32_t N,
        ms_image r_img,
        target_rect r_rect,
        ms_image* imgs,
        target_rect* img_rects,
        double* fused_data64_t,
        unsigned char* fused_data8_t,
        bool tight_box
    );


#ifdef __cplusplus
}
#endif

#endif  // _TURBULENCE_SIMULATION_LIBRARY_H_
