#ifndef PARAMENT_CONTEXT_H_
#define PARAMENT_CONTEXT_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#define NO_CUDA_STUBS
#include "parament.h"


struct Parament_Context {
    bool initialized;
    
    // Handles
    cublasHandle_t cublasHandle;

    // GPU Arrays and constants
    cuComplex* H0;
    cuComplex* H1;
    cuComplex* one_GPU;
    cuComplex *one_GPU_diag;

    // Dimension of the Hilbert space
    unsigned int dim;

    // Currently initialized time steps
    unsigned int curr_max_pts;

    // Point arrays
    cuComplex* c0;
    cuComplex* c1;
    cuComplex* X;
    cuComplex* D0;
    cuComplex* D1;

    // check vars
    bool hamiltonian_is_set = false;

    // BESSEL COEFFICIENTS
    cuComplex* J;
    unsigned int MMAX;
    float alpha;
    float beta;

    // Device 
    int numSMs;

    enum Parament_ErrorCode lastError;
};
#endif  // PARAMENT_CONTEXT_H_
