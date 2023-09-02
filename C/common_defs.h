/**
 * @file
 **/

#ifndef VA_COMMON_DEFS_H
#define VA_COMMON_DEFS_H

// use 64-bit representation of real numbers
#define VA_REAL double
// use 32-bit representation of real numbers
//#define VA_REAL float

#define VA_PI 3.14159265358979323846

#if defined(CUDA)
    #include <cuda/std/complex>
    #define VA_DEVICE_ADDR
    #define VA_DEVICE_FUN __device__
    #define fabs fabsf
    #define sqrt sqrtf
    #define pow powf
    #define VA_COMPLEX cuda::std::complex<VA_REAL>
#elif defined(OPENCL)
    #define VA_DEVICE_ADDR __global
    #define VA_DEVICE_FUN
    #define sqrt native_sqrt
    #define sin native_sin
    #define cos native_cos
    #define VA_COMPLEX complex VA_REAL
#else
    #include <math.h>
    #include <stdlib.h>
    #include <stdio.h>
    #include <complex.h>
    #include <float.h>
    #define VA_DEVICE_ADDR
    #define VA_DEVICE_FUN
    #define VA_COMPLEX complex VA_REAL
    #define VA_INFINITY FLT_MAX
#endif

#endif // VA_COMMON_DEFS_H
