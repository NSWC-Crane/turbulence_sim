#ifndef _LENS_PIXEL_SIZE_H_
#define _LENS_PIXEL_SIZE_H_

#include <cstdint>
#include <cmath>

//-----------------------------------------------------------------------------
// this function returns the curve fit of real data for a given lens zoom setting and range (L)
inline double get_pixel_size(uint32_t zoom, double range)
{
    return (-0.003657 + (4.707e-07 * zoom) + (1.791e-05 * range) + (-1.893e-11 * zoom * zoom) + (-1.84e-09 * zoom * range) + (-1.378e-08 * range * range) //
        + (7.507e-14 * zoom * zoom * range) + (6.222e-13 * zoom * range * range) + (5.017e-12 * range * range * range));
}   // end of get_pixel_size

#endif  // _LENS_PIXEL_SIZE_H_
