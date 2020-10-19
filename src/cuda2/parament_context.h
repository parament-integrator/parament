#ifndef PARAMENT_CONTEXT_H_
#define PARAMENT_CONTEXT_H_

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdbool.h>

#define NO_CUDA_STUBS
#include "parament.h"


struct Parament_Context {
    bool initialized;
    
    // Handles
    cublasHandle_t cublasHandle;

    // GPU Arrays and constants
    cuComplex *H0;
    cuComplex *H1;
    cuComplex *one_GPU;
    cuComplex *one_GPU_diag;

    // Dimension of the Hilbert space
    unsigned int dim;
    
    // Hnorm used for automatically setting the # of iteration cycles
    float Hnorm;

    // Currently initialized time steps
    unsigned int curr_max_pts;

    // Point arrays
    cuComplex *c0;
    cuComplex *c1;
    cuComplex *X;
    cuComplex *D0;
    cuComplex *D1;

    // check vars
    bool hamiltonian_is_set;

    // BESSEL COEFFICIENTS
    cuComplex *J;
    float alpha;
    float beta;

    // Device 
    int numSMs;

    enum Parament_ErrorCode lastError;

    // Commonly used constants
    // (can't be actually const, because we compute the value by calling make_complex())
    cuComplex zero;
    cuComplex one;
    cuComplex two;
    cuComplex mone;
    cuComplex mtwo;

    // Iteration cycle number
    unsigned int MMAX;
    bool MMAX_manual;

};
#endif  // PARAMENT_CONTEXT_H_
