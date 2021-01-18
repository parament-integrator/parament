#ifndef PARAMENT_CONTEXT_H_
#define PARAMENT_CONTEXT_H_

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdbool.h>

#define NO_CUDA_STUBS
#include "parament.h"


template<typename complex_t> struct Parament_Context {
    // Handles
    cublasHandle_t cublasHandle;

    // GPU Arrays and constants
    complex_t *H0;
    complex_t *H1;
    complex_t *one_GPU;
    complex_t *one_GPU_diag;

    // Dimension of the Hilbert space
    unsigned int dim;
    
    // Hnorm used for automatically setting the # of iteration cycles
    float Hnorm;

    // Currently allocated time steps
    unsigned int curr_max_pts;

    // Point arrays
    complex_t *c0;
    complex_t *c1;
    complex_t *X;
    complex_t *D0;
    complex_t *D1;

    // BESSEL COEFFICIENTS
    complex_t *J;
    float alpha;
    float beta;

    // Device 
    int numSMs;

    enum Parament_ErrorCode lastError;

    // Commonly used constants
    // (can't be actually const, because we compute the value by calling make_complex())
    complex_t zero;
    complex_t one;
    complex_t two;
    complex_t mone;
    complex_t mtwo;

    // Iteration cycle number
    unsigned int MMAX;
    bool MMAX_manual;
};

#endif  // PARAMENT_CONTEXT_H_
