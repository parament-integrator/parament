/* Copyright 2021 Konstantin Herb, Pol Welter. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


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
    double Hnorm;

    // Currently allocated time steps
    unsigned int curr_max_pts;

    // Point arrays
    complex_t *c0;
    complex_t *c1;
    complex_t *c2;
    complex_t *X;
    complex_t *D0;
    complex_t *D1;

    // BESSEL COEFFICIENTS
    complex_t *J;
    double alpha;
    double beta;

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

    // Integration parameters
    bool enable_magnus;
    Parament_QuadratureSpec quadrature_mode;
};

#endif  // PARAMENT_CONTEXT_H_
