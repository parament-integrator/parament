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


#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include "parament_context.hpp"
#include "diagonal_add.h"
#include "mathhelper.h"
#include "control_expansion.h"
#include "parament_blas.hpp"
#include "debugfuncs.hpp"

#define ENDLINE "\n"

#ifndef NDEBUG
    #define PARAMENT_DEBUG(...) {\
        fprintf(stderr, "%s", "PARAMENT_DEBUG: ");\
        fprintf(stderr, __VA_ARGS__);\
        fprintf(stderr, ENDLINE);\
    }

    #define PARAMENT_ERROR(...) {\
        fprintf(stderr, "%s", "PARAMENT_ERROR: ");\
        fprintf(stderr, __VA_ARGS__);\
        fprintf(stderr, ENDLINE);\
    }
#else
    #define PARAMENT_DEBUG(...) ((void)0)
    #define PARAMENT_ERROR(...) ((void)0)
#endif



/*
 * Create Parment context and prepare GPU
 */
template<typename complex_t>
Parament_ErrorCode Parament_create(Parament_Context<complex_t> **handle_p) {
    // todo (Pol): use `new` operator instead of `malloc`. Casting `malloc` is BAD.
    auto *handle = reinterpret_cast<Parament_Context<complex_t>*>(malloc(sizeof(Parament_Context<complex_t>)));
    *handle_p = handle;
    if (handle == NULL) {
        return PARAMENT_STATUS_HOST_ALLOC_FAILED;
    }

    handle->H0 = NULL;
    handle->H1 = NULL;
    handle->one_GPU = NULL;
    handle->one_GPU_diag = NULL;
    handle->c0 = NULL;
    handle->c1 = NULL;
    handle->X = NULL;
    handle->D0 = NULL;
    handle->D1 = NULL;
    handle->J = NULL;

    // initialize options
    handle->MMAX = 11;
    handle->MMAX_manual = false;
    handle->quadrature_mode = PARAMENT_QUADRATURE_SIMPSON;
    handle->enable_magnus = true;

    // BESSEL COEFFICIENTS
    handle->alpha = -2.0;
    handle->beta = 2.0;

    // Commonly used constants
    handle->zero = makeComplex<complex_t>(0,0);
    handle->one = makeComplex<complex_t>(1,0);
    handle->two = makeComplex<complex_t>(2,0);
    handle->mone = makeComplex<complex_t>(-1,0);
    handle->mtwo = makeComplex<complex_t>(-2,0);

    cudaError_t error = cudaSuccess;
    cublasStatus_t cublasError = CUBLAS_STATUS_SUCCESS;

    // initialize cublas context
    if (CUBLAS_STATUS_SUCCESS != cublasCreate(&(handle->cublasHandle))) {
        handle->lastError = PARAMENT_STATUS_CUBLAS_INIT_FAILED;
        goto error_cleanup1;
    }

    // initialize device memory
    if (cudaSuccess != cudaMalloc(&handle->one_GPU, sizeof(complex_t))) {
        handle->lastError = PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        goto error_cleanup2;
    }
    
    error = cudaMemcpy(handle->one_GPU, &handle->one, sizeof(complex_t), cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);

    handle->curr_max_pts = 0; // No points yet allocated

    if (cudaSuccess != cudaDeviceGetAttribute(&handle->numSMs, cudaDevAttrMultiProcessorCount, 0)) {
        // TODO (Pol): better error code here. also, make the GPU configurable
        handle->lastError = PARAMENT_FAIL;
        goto error_cleanup3;
    }
    PARAMENT_DEBUG("created Parament context");
    PARAMENT_DEBUG("%s", typeid(complex_t).name());
    handle->lastError = PARAMENT_STATUS_SUCCESS;
    return handle->lastError;

error_cleanup3:
    error = cudaFree(handle->one_GPU);
    assert(error == cudaSuccess);
error_cleanup2:
    cublasError = cublasDestroy(handle->cublasHandle);
    assert(cublasError == CUBLAS_STATUS_SUCCESS);
error_cleanup1:
    free(handle);
    *handle_p = NULL;
    return handle->lastError;
}

/*
 * Frees a previously allocated hamiltionian. No-op if no hamiltonian has been allocated.
 */
template<typename complex_t>
static void freeHamiltonian(Parament_Context<complex_t> *handle) {
    cudaError_t error;
    PARAMENT_DEBUG("Freeing handle->H0 = 0x%p", handle->H0);
    error = cudaFree(handle->H0);
    assert(error == cudaSuccess);
    PARAMENT_DEBUG("Freeing handle->H1 = 0x%p", handle->H1);
    error = cudaFree(handle->H1);
    assert(error == cudaSuccess);
    PARAMENT_DEBUG("Freeing handle->one_GPU_diag = 0x%p", handle->one_GPU_diag);
    error = cudaFree(handle->one_GPU_diag);
    assert(error == cudaSuccess);

    handle->H0 = NULL;
    handle->H1 = NULL;
    handle->one_GPU_diag = NULL;
}

/*
 * Frees a previously allocated control field and working memory. No-op if no control field has been allocated.
 */
template<typename complex_t>
static void freeWorkingMemory(Parament_Context<complex_t> *handle) {
    if (handle->curr_max_pts != 0) {
    cudaError_t error;
    PARAMENT_DEBUG("Freeing handle->c0: 0x%p", handle->c0);
    error = cudaFree(handle->c0);
    assert(error == cudaSuccess);
    PARAMENT_DEBUG("Freeing handle->c1: 0x%p", handle->c1);
    error = cudaFree(handle->c1);
    assert(error == cudaSuccess);
    PARAMENT_DEBUG("Freeing handle->X: 0x%p", handle->X);
    error = cudaFree(handle->X);
    assert(error == cudaSuccess);
    PARAMENT_DEBUG("Freeing handle->D0: 0x%p", handle->D0);
    error = cudaFree(handle->D0);
    assert(error == cudaSuccess);
    PARAMENT_DEBUG("Freeing handle->D1: 0x%p", handle->D1);
    error = cudaFree(handle->D1);
    assert(error == cudaSuccess);
    handle->c0 = NULL;
    handle->c1 = NULL;
    handle->X = NULL;
    handle->D0 = NULL;
    handle->D1 = NULL;
    handle->curr_max_pts = 0;
    }
}

/*
 * Destructor for the Parament context
 */
template<typename complex_t>
Parament_ErrorCode Parament_destroy(Parament_Context<complex_t> *handle) {
    cudaError_t cudaError;
    cudaError = cudaDeviceSynchronize();
    assert(cudaError == cudaSuccess);
    if (NULL == handle)
        return PARAMENT_STATUS_SUCCESS;

    PARAMENT_DEBUG("Freeing J = 0x%p", handle->J);
    free(handle->J);
    PARAMENT_DEBUG("Freeing handle->one_GPU = 0x%p", handle->one_GPU);
    cudaError = cudaFree(handle->one_GPU);
    assert(cudaError == cudaSuccess);
    PARAMENT_DEBUG("destroying cublasHandle = 0x%p", handle->cublasHandle);
    cublasStatus_t cublasStatus = cublasDestroy(handle->cublasHandle);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    freeHamiltonian(handle);
    freeWorkingMemory(handle);
    free(handle);
    return PARAMENT_STATUS_SUCCESS;
}

/*
 * Upload the system Hamiltonians to the GPU. The routines also sets the quadrature settings used for the integration
 */
template<typename complex_t>
Parament_ErrorCode Parament_setHamiltonian(
        Parament_Context<complex_t> *handle, complex_t *H0, complex_t *H1, unsigned int dim, unsigned int amps,
        bool use_magnus, Parament_QuadratureSpec quadrature_mode) {
    
    // Hamiltonian might have been set before, deallocate first
    freeHamiltonian(handle);

    handle->dim = dim;

    if (use_magnus) {
        PARAMENT_DEBUG("Magnus enabled");
        handle->enable_magnus = true;
        if (quadrature_mode != PARAMENT_QUADRATURE_SIMPSON){
            handle->lastError = PARAMENT_STATUS_INVALID_QUADRATURE_SELECTION;
            goto error_cleanup;  
        }
        handle->quadrature_mode = PARAMENT_QUADRATURE_SIMPSON;
    }
    else
    {
        handle->enable_magnus = false;
        handle->quadrature_mode = quadrature_mode;
    }

    int H1memory;

    if (use_magnus) {
        // amps*(amps-1)/2 pairwise commutators + amps commutators with H0 + amps "real control Hamiltonians"
        H1memory = dim * dim * sizeof(complex_t) * ( 2*amps + (amps*(amps-1))/2 );

    }
    else
    {
        H1memory = dim * dim * amps * sizeof(complex_t);
    }
    
    // Allocate GPU memory
    if (cudaSuccess != cudaMalloc(&handle->H0, dim * dim * sizeof(complex_t))) {
        handle->lastError = PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        goto error_cleanup;
    }

    if (cudaSuccess != cudaMalloc(&handle->H1, H1memory)) {
        handle->lastError = PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        goto error_cleanup;
    }

    

    // Transfer to GPU
    cudaError_t error;
    error = cudaMemcpy(handle->H0, H0, dim * dim * sizeof(complex_t), cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);

    error = cudaMemcpy(handle->H1, H1, dim*dim*amps*sizeof(complex_t), cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);

    // Helper Arrays
    if (cudaSuccess != cudaMalloc(&handle->one_GPU_diag, dim * sizeof(complex_t))) {
        handle->lastError = PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        goto error_cleanup;
    }
    if (CUBLAS_STATUS_SUCCESS != cublasGaxpy(handle->cublasHandle, dim, &handle->one, handle->one_GPU, 0,
            handle->one_GPU_diag, 1)) {
        handle->lastError = PARAMENT_STATUS_CUBLAS_FAILED;
        goto error_cleanup;
    }

    PARAMENT_DEBUG("Copied to GPU");

    // Determine norm. We use the 1-norm as an upper limit
    handle->Hnorm = OneNorm(H0,dim);
    for (int i = 0; i < amps; i++){
        PARAMENT_DEBUG("Iterate over amplitude");
         handle->Hnorm += OneNorm(H1+i*dim*dim,dim);
    }
    
    handle->alpha = -handle->Hnorm;
    handle->beta = handle->Hnorm;
    
    if (use_magnus){
        PARAMENT_DEBUG("Calculate Commutators");
        complex_t *currentH1Addr;
        complex_t *currentH2Addr;
        complex_t *currentCommAddr;

        for (int i = 0;i<amps;++i){

           
            
            currentH1Addr   = handle->H1 + i*dim*dim;
            currentCommAddr = handle->H1 + (i+amps)*dim*dim;

            // Commutators with H0
            if (CUBLAS_STATUS_SUCCESS != cublasGgemm(handle->cublasHandle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim, dim, dim,
            &handle->one,
            handle->H0, dim,
            currentH1Addr, dim,
            &handle->zero,
            currentCommAddr, dim)){
                handle->lastError = PARAMENT_STATUS_CUBLAS_FAILED;
                goto error_cleanup;
            };

            if (CUBLAS_STATUS_SUCCESS != cublasGgemm(handle->cublasHandle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim, dim, dim,
            &handle->mone,
            currentH1Addr, dim,
            handle->H0, dim,
            &handle->one,
            currentCommAddr, dim)){
                handle->lastError = PARAMENT_STATUS_CUBLAS_FAILED;
                goto error_cleanup;
            };


            for (int j = 0;j<amps;++j){
                if (j<i){
                // Pairwise commutators
                currentH1Addr   = handle->H1 + i*dim*dim;
                currentH2Addr   = handle->H1 + j*dim*dim;
                currentCommAddr = handle->H1 + (2*amps+(i-1)+j)*dim*dim;



                if (CUBLAS_STATUS_SUCCESS != cublasGgemm(handle->cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim, dim, dim,
                &handle->one,
                currentH2Addr, dim,
                currentH1Addr, dim,
                &handle->zero,
                currentCommAddr, dim)){
                    handle->lastError = PARAMENT_STATUS_CUBLAS_FAILED;
                    goto error_cleanup;
                };

                if (CUBLAS_STATUS_SUCCESS != cublasGgemm(handle->cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim, dim, dim,
                &handle->mone,
                currentH1Addr, dim,
                currentH2Addr, dim,
                &handle->one,
                currentCommAddr, dim)){
                    handle->lastError = PARAMENT_STATUS_CUBLAS_FAILED;
                    goto error_cleanup;
                };
                }
                
            }
        }
    }

    // nvtxMarkA("Set Hamiltonian routine completed");
    handle->lastError = PARAMENT_STATUS_SUCCESS;
    return handle->lastError;

error_cleanup:
    freeHamiltonian(handle);
    return handle->lastError;
}

/*
 * Computes the Bessel coefficients for the integration
 */
template<typename complex_t, typename real_t>
static Parament_ErrorCode equipropComputeCoefficients(Parament_Context<complex_t> *handle, real_t dt) {
    // If enabled, automatically determine number of iterations
    if (!handle->MMAX_manual){
        int MMAX_selected = -1;
        if (typeid(complex_t) == typeid(cuComplex)) {
            MMAX_selected = Parament_selectIterationCycles_fp32(handle->Hnorm, dt);
            PARAMENT_DEBUG("The FP32 selectIterationCycles routine has taken %d", MMAX_selected);
        }
        if (typeid(complex_t) == typeid(cuDoubleComplex)) {
            MMAX_selected = Parament_selectIterationCycles_fp64(handle->Hnorm, dt);
            PARAMENT_DEBUG("The FP64 selectIterationCycles routine has taken %d", MMAX_selected);
        }
        if (MMAX_selected < 3){
            return PARAMENT_STATUS_SELECT_SMALLER_DT;
        }

        handle->MMAX = MMAX_selected;
    }
    PARAMENT_DEBUG("\nMMAX is %d\n",handle->MMAX);

    // Allocate Bessel coefficients
    free(handle->J);
    handle->J = NULL;
    // todo (Pol): use `new` operator instead of `malloc`. Casting `malloc` is BAD.
    handle->J = reinterpret_cast<complex_t*>(malloc(sizeof(complex_t) * (handle->MMAX + 1)));
    if (handle->J == NULL) {
        return PARAMENT_STATUS_HOST_ALLOC_FAILED;
    }

    // Compute Bessel coeffficients
    double x = dt*(handle->beta - handle->alpha)/2;
    for (int k = 0; k < handle->MMAX + 1; k++) {
        handle->J[k] = calculate_bessel_coeffs<complex_t>(k,x);
    }
    return PARAMENT_STATUS_SUCCESS;
}


/*
 * Transfer arrays of the control amplitudes to the GPU
 */
template<typename complex_t>
static Parament_ErrorCode equipropTransfer(Parament_Context<complex_t> *handle, complex_t *carr, unsigned int pts, unsigned int amps) {
    // Allocate memory for c arrays if needed
    if (handle->curr_max_pts < pts) {
        PARAMENT_DEBUG("Need to free arrays on GPU");
        freeWorkingMemory(handle);

        PARAMENT_DEBUG("Need to malloc arrays on GPU");
        unsigned int dim = handle-> dim;

        if (cudaSuccess != cudaMalloc(&handle->c0, pts * sizeof(complex_t))
                || cudaSuccess != cudaMalloc(&handle->c1, pts * amps * sizeof(complex_t))
                || cudaSuccess != cudaMalloc(&handle->X, dim * dim * pts * sizeof(complex_t))
                || cudaSuccess != cudaMalloc(&handle->D0, dim * dim * pts * sizeof(complex_t))
                || cudaSuccess != cudaMalloc(&handle->D1, dim * dim * pts * sizeof(complex_t))) {
            freeWorkingMemory(handle);
            return PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        }

        // If magnus enabled, then get a c2 array of appropriate size encorpoarting the commutators
        if (handle->enable_magnus == true) {
                PARAMENT_DEBUG("Malloc c2 for Magnus");
                if (cudaSuccess != cudaMalloc(&handle->c2, (pts-1)/2*(2*amps+(amps*(amps-1))/2) * sizeof(complex_t))) {
                    freeWorkingMemory(handle);
                    return PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
                    } 
        }
       
        // If quadrature enabled
        if ((handle->quadrature_mode == PARAMENT_QUADRATURE_MIDPOINT) && (handle->enable_magnus == false)) {
            PARAMENT_DEBUG("Malloc c2 for Midpoint");
            if (cudaSuccess != cudaMalloc(&handle->c2, (pts-1)*amps * sizeof(complex_t))) {
                freeWorkingMemory(handle);
                return PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
                } 
        }

        if ((handle->quadrature_mode == PARAMENT_QUADRATURE_SIMPSON) && (handle->enable_magnus == false)){
            PARAMENT_DEBUG("Malloc c2 for Simpson");
            if (cudaSuccess != cudaMalloc(&handle->c2, (pts-1)/2*amps * sizeof(complex_t))) {
                freeWorkingMemory(handle);
                return PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
                } 
        }

    
        // Memorize how many pts are initalized
        handle->curr_max_pts = pts;

        // Fill c0 array with ones
        if (CUBLAS_STATUS_SUCCESS != cublasGscal(handle->cublasHandle, pts, &handle->zero, handle->c0, 1)) {
            freeWorkingMemory(handle);
            return PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        }
        if (CUBLAS_STATUS_SUCCESS != cublasGaxpy(handle->cublasHandle, pts, &handle->one, handle->one_GPU, 0,
                handle->c0, 1)) {
            freeWorkingMemory(handle);
            return PARAMENT_STATUS_CUBLAS_FAILED;
        }
    }

    // Transfer c1
    cudaError_t error = cudaMemcpy(handle->c1, carr, amps * pts * sizeof(complex_t), cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);

    return PARAMENT_STATUS_SUCCESS;
}

/*
 * Outlaying of all slice Hamiltonians / effective Hamiltonians for subsequent exponentiation
 */
template<typename complex_t>
static Parament_ErrorCode equipropExpand(Parament_Context<complex_t> *handle, unsigned int pts, unsigned int amps, double dt) {

    unsigned int dim = handle->dim;
    cublasStatus_t error;
    error = cublasGgemm(handle->cublasHandle,
         CUBLAS_OP_N, CUBLAS_OP_N,
         dim*dim, pts, 1,
         &handle->one,
         handle->H0, dim*dim,
         handle->c0, 1,
         &handle->zero,
         handle->X, dim*dim);
    if (error != CUBLAS_STATUS_SUCCESS) {
        return PARAMENT_STATUS_CUBLAS_FAILED;
    }


    // Midpoint, Simpson, Magnus

    complex_t* expansion_array;
    unsigned int expansion_pts;
    unsigned int expansion_amps;

    if ((handle->quadrature_mode == PARAMENT_QUADRATURE_NONE) && (handle->enable_magnus == false)){
        expansion_array = handle->c1;
        expansion_amps = amps;
        expansion_pts = pts;
    }

    if ((handle->quadrature_mode == PARAMENT_QUADRATURE_MIDPOINT) && (handle->enable_magnus == false)){
        // Kernel launch for midpoint
        control_midpoint(handle->c1,handle->c2,amps,pts,handle->numSMs);        
        expansion_array = handle->c2;
        expansion_amps = amps;
        expansion_pts = pts-1;
    }

    if ((handle->quadrature_mode == PARAMENT_QUADRATURE_SIMPSON) && (handle->enable_magnus == false)){
        // Kernel launch for Simpson
        control_simpson(handle->c1,handle->c2,amps,pts,handle->numSMs);
        expansion_array = handle->c2;
        expansion_amps = amps;
        expansion_pts = (pts-1)/2;
    }

    if (handle->enable_magnus == true){
        control_magnus(handle->c1,handle->c2,amps,pts,dt, handle->numSMs);
        expansion_array = handle->c2;
        expansion_amps = 2*amps+((amps-1)*amps)/2;
        expansion_pts = (pts-1)/2;

    }


    error = cublasGgemm(handle->cublasHandle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim*dim, expansion_pts, expansion_amps,
        &handle->one,
        handle->H1, dim*dim,
        expansion_array, expansion_pts,
        &handle->one,
        handle->X, dim*dim);
    if (error != CUBLAS_STATUS_SUCCESS) {
        return PARAMENT_STATUS_CUBLAS_FAILED;
    }  

    return PARAMENT_STATUS_SUCCESS;
}


/*
 * Exponentiation of the time slices
 */
template<typename complex_t>
static Parament_ErrorCode equipropPropagate(Parament_Context<complex_t> *handle, double dt, unsigned int pts) {
    // define some short-form aliases...
    const unsigned int dim = handle->dim;
    complex_t *const D0 = handle->D0;
    complex_t *const D1 = handle->D1;
    complex_t *const X = handle->X;

    cublasStatus_t error;

    // Rescale dt
    dt = 2/(handle->beta - handle->alpha)*2;
    complex_t dt_complex = makeComplex<complex_t>(dt, 0);

    complex_t* ptr_accumulate;

    for (int k = handle->MMAX; k >= 0; k--) {
        if (k == handle->MMAX){
            error = cublasGscal(handle->cublasHandle, pts*dim*dim, &handle->zero, D0, 1);
            if (error != CUBLAS_STATUS_SUCCESS)
                return PARAMENT_STATUS_CUBLAS_FAILED;
        }
        else {
            // D0 = D0 + 2 X @ D1 * dt
            error = cublasGgemmStridedBatched(handle->cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim, dim, dim,
                &dt_complex,
                X, dim, dim*dim,
                D1, dim, dim*dim,
                &handle->mone,
                D0, dim, dim*dim,
                pts
            );
            if (error != CUBLAS_STATUS_SUCCESS)
                return PARAMENT_STATUS_CUBLAS_FAILED;
        }

        // D0 = D0 + I*ak
        assert(cudaPeekAtLastError() == cudaSuccess);
        diagonal_add(handle->J[k], D0, pts, handle->numSMs, handle->dim);
        assert(cudaPeekAtLastError() == cudaSuccess);

        // Next step
        k--;

        if (k == handle->MMAX - 1) {
            ptr_accumulate = &handle->zero;
            //cublasGscal(handle, pts*dim*dim, &zero, D1, 1);
        }
        if (k == 0){
            ptr_accumulate = &handle->mtwo;
        }

        // D1 = D1 + 2 X @ D0
        error = cublasGgemmStridedBatched(handle->cublasHandle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim, dim, dim,
            &dt_complex,
            X, dim, dim*dim,
            D0, dim, dim*dim,
            ptr_accumulate,
            D1, dim, dim*dim,
            pts
        );
        if (error != CUBLAS_STATUS_SUCCESS)
            return PARAMENT_STATUS_CUBLAS_FAILED;

       // D1 = D1 + I*ak'
       assert(cudaPeekAtLastError() == cudaSuccess);
       diagonal_add(handle->J[k], D1, pts, handle->numSMs, handle->dim);
       assert(cudaPeekAtLastError() == cudaSuccess);

       if (k == handle->MMAX - 1){
           ptr_accumulate = &handle->mone;
       }
    }
    // D1 contains now the matrix exponentials
    return PARAMENT_STATUS_SUCCESS;
}

/*
 * Multipy all matrix exponentials together
 */
template<typename complex_t>
static Parament_ErrorCode equipropReduce(Parament_Context<complex_t> *handle, unsigned int pts) {
    // define some short-form aliases...
    const unsigned int dim = handle->dim;
    complex_t *const D1 = handle->D1;

    //readback(D1,dim*dim*pts);

    cublasStatus_t error;

    int remain_pts = pts;
    int pad = 0;
    while (remain_pts > 1){
        pad = remain_pts % 2;
        remain_pts = remain_pts/2;

        error = cublasGgemmStridedBatched(handle->cublasHandle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim, dim, dim,
            &handle->one,
            D1          , dim, dim*dim*2,
            D1 + dim*dim, dim, dim*dim*2,
            &handle->zero,
            D1, dim, dim*dim,
            remain_pts
        );
        if (error != CUBLAS_STATUS_SUCCESS)
            return PARAMENT_STATUS_CUBLAS_FAILED;

        if (pad>0){
            // One left over, need to copy to Array
            error = cublasGcopy(handle->cublasHandle,
                dim*dim,
                D1 + dim*dim*(remain_pts*2), 1,
                D1 + dim*dim*(remain_pts), 1
            );
            if (error != CUBLAS_STATUS_SUCCESS)
                return PARAMENT_STATUS_CUBLAS_FAILED;
            remain_pts += 1;
        }
    }
    return PARAMENT_STATUS_SUCCESS;
}

/*
 * Look-up for norm threasholds for FP32
 */
int Parament_selectIterationCycles_fp32(double H_norm, double dt) {
    PARAMENT_DEBUG("FP32 HNORM*DT = %f\n", H_norm*dt );
    if (H_norm*dt <= 0.032516793) { return 3; };
    if (H_norm*dt <= 0.219062571) { return 5; };
    if (H_norm*dt <= 0.619625593) { return 7; };
    if (H_norm*dt <= 1.218059203) { return 9; };
    if (H_norm*dt <= 1.979888284) { return 11; };
    if (H_norm*dt <= 2.873301187) { return 13; };
    if (H_norm*dt <= 3.872963682) { return 15; };
    if (H_norm*dt <= 4.959398466) { return 17; };
    if (H_norm*dt <= 6.117657121) { return 19; };
    if (H_norm*dt <= 7.336154907) { return 21; };
    if (H_norm*dt <= 8.605792444) { return 23; };
    if (H_norm*dt <= 9.919320831) { return 25; };
    if (H_norm*dt <= 11.27088616) { return 27; };
    if (H_norm*dt <= 12.65570085) { return 29; } else {return -1;};
}

/*
 * Look-up for norm threasholds for FP64
 */
int Parament_selectIterationCycles_fp64(double H_norm, double dt) {
    PARAMENT_DEBUG("HNORM*DT = %f\n", H_norm*dt );
    if (H_norm*dt <= 0.000213616) { return 3; };
    if (H_norm*dt <= 0.00768149) { return 5; };
    if (H_norm*dt <= 0.0501474) { return 7; };
    if (H_norm*dt <= 0.162592) { return 9; };
    if (H_norm*dt <= 0.368382) { return 11; };
    if (H_norm*dt <= 0.676861) { return 13; };
    if (H_norm*dt <= 1.08784) { return 15; };
    if (H_norm*dt <= 1.59605) { return 17; };
    if (H_norm*dt <= 2.19402) { return 19; };
    if (H_norm*dt <= 2.87366) { return 21; };
    if (H_norm*dt <= 3.62716) { return 23; };
    if (H_norm*dt <= 4.44725) { return 25; };
    if (H_norm*dt <= 5.3274)  { return 27; };
    if (H_norm*dt <= 6.26179) { return 29; };
    if (H_norm*dt <= 7.2453)  { return 31; };
    if (H_norm*dt <= 8.27338) { return 33; };
    if (H_norm*dt <= 9.34206) { return 35; };
    if (H_norm*dt <= 10.4478) { return 37; };
    if (H_norm*dt <= 11.5875) { return 39; };
    if (H_norm*dt <= 12.7584) { return 41; } else {return -1;};
}

/*
 * Set Chebychev iterations manually
 */
template<typename complex_t>
Parament_ErrorCode Parament_setIterationCyclesManually(Parament_Context<complex_t> *handle, unsigned int cycles) {
    handle->MMAX = cycles;
    handle->MMAX_manual = true;
    return PARAMENT_STATUS_SUCCESS;
}

/*
 * Enable dynmic Checbychev cycle selection
 */
template<typename complex_t>
Parament_ErrorCode Parament_automaticIterationCycles(Parament_Context<complex_t> *handle) {
    handle->MMAX = 11;
    handle->MMAX_manual = false;
    return PARAMENT_STATUS_SUCCESS;
}

/*
 * Main integration routine. This routine performs all steps and is exported.
 */
template<typename complex_t>
Parament_ErrorCode Parament_equiprop(Parament_Context<complex_t> *handle, complex_t *carr, double dt, unsigned int pts, unsigned int amps, complex_t *out) {
    PARAMENT_DEBUG("Equiprop C called");
    if (handle->H0 == NULL) {
        handle->lastError = PARAMENT_STATUS_NO_HAMILTONIAN;
        return handle->lastError;
    }

    if ((handle->enable_magnus == true) || (handle->quadrature_mode == PARAMENT_QUADRATURE_SIMPSON)){
        dt = 2*dt;
    }

    handle->lastError = equipropComputeCoefficients(handle, dt);
    if (PARAMENT_STATUS_SUCCESS != handle->lastError) {
        return handle->lastError;
    }
    PARAMENT_DEBUG("Finished Computation of coefficients");

    handle->lastError = equipropTransfer(handle, carr, pts, amps);
    if (PARAMENT_STATUS_SUCCESS != handle->lastError) {
        return handle->lastError;
    }

    handle->lastError = equipropExpand(handle, pts, amps, dt);
    if (PARAMENT_STATUS_SUCCESS != handle->lastError) {
        return handle->lastError;
    }

    handle->lastError = equipropPropagate(handle, dt, pts);
    if (PARAMENT_STATUS_SUCCESS != handle->lastError) {
        return handle->lastError;
    }

    handle->lastError = equipropReduce(handle, pts);
    if (PARAMENT_STATUS_SUCCESS != handle->lastError) {
        return handle->lastError;
    }

    // transfer back
    const unsigned int dim = handle->dim;
    complex_t *const D1 = handle->D1;
    cudaError_t cudaError = cudaMemcpy(out, D1, dim * dim  * sizeof(complex_t), cudaMemcpyDeviceToHost);
    assert(cudaError == cudaSuccess);

    handle->lastError = PARAMENT_STATUS_SUCCESS;
    return PARAMENT_STATUS_SUCCESS;
}


template<typename complex_t>
Parament_ErrorCode Parament_peekAtLastError(Parament_Context<complex_t> *handle) {
    return handle->lastError;
}

const char *Parament_errorMessage(Parament_ErrorCode errorCode) {
    switch (errorCode) {
    case PARAMENT_STATUS_SUCCESS:
        return "Success";
    case PARAMENT_STATUS_HOST_ALLOC_FAILED:
        return "Memory allocation on the host failed.";
    case PARAMENT_STATUS_DEVICE_ALLOC_FAILED:
        return "Memory allocation on the device failed.";
    case PARAMENT_STATUS_CUBLAS_INIT_FAILED:
        return "Failed to initialize the cuBLAS library.";
    case PARAMENT_STATUS_INVALID_VALUE:
        return "Invalid value.";
    case PARAMENT_STATUS_CUBLAS_FAILED:
        return "Failed to execute cuBLAS function.";
    case PARAMENT_STATUS_SELECT_SMALLER_DT:
        return "Timestep too large";
    case PARAMENT_STATUS_INVALID_QUADRATURE_SELECTION:
        return "Invalid quadrature selection";
    default:
        return "Unknown error code";
    }
}



// ======================================================================
// Implementation of the actually exported (i.e. non-templated) functions
// ======================================================================

Parament_ErrorCode Parament_create(Parament_Context_f32 **handle_p) {
    return Parament_create<cuComplex>(reinterpret_cast<Parament_Context<cuComplex>**>(handle_p));
}

Parament_ErrorCode Parament_destroy(Parament_Context_f32 *handle) {
    return Parament_destroy<cuComplex>(reinterpret_cast<Parament_Context<cuComplex>*>(handle));
}

Parament_ErrorCode Parament_setHamiltonian(Parament_Context_f32 *handle, cuComplex *H0, cuComplex *H1,
        unsigned int dim, unsigned int amps, bool use_magnus, Parament_QuadratureSpec quadrature_mode) {
    return Parament_setHamiltonian<cuComplex>(reinterpret_cast<Parament_Context<cuComplex>*>(handle), H0, H1, dim, amps, use_magnus, quadrature_mode);
}

Parament_ErrorCode Parament_equiprop(Parament_Context_f32 *handle, cuComplex *carr, double dt, unsigned int pts,
        unsigned int amps, cuComplex *out) {
    return Parament_equiprop<cuComplex>(reinterpret_cast<Parament_Context<cuComplex>*>(handle), carr, dt, pts, amps, out);
}

Parament_ErrorCode Parament_setIterationCyclesManually(Parament_Context_f32 *handle, unsigned int cycles) {
    return Parament_setIterationCyclesManually<cuComplex>(reinterpret_cast<Parament_Context<cuComplex>*>(handle),
        cycles);
}

Parament_ErrorCode Parament_automaticIterationCycles(Parament_Context_f32 *handle) {
    return Parament_automaticIterationCycles<cuComplex>(reinterpret_cast<Parament_Context<cuComplex>*>(handle));
}

Parament_ErrorCode Parament_peekAtLastError(Parament_Context_f32 *handle) {
    return Parament_peekAtLastError<cuComplex>(reinterpret_cast<Parament_Context<cuComplex>*>(handle));
}

// ======================================================================
// Implementation of the actually exported (i.e. non-templated) functions
// ======================================================================

Parament_ErrorCode Parament_create_fp64(Parament_Context_f64 **handle_p) {
    return Parament_create<cuDoubleComplex>(reinterpret_cast<Parament_Context<cuDoubleComplex>**>(handle_p));
}

Parament_ErrorCode Parament_destroy_fp64(Parament_Context_f64 *handle) {
    return Parament_destroy<cuDoubleComplex>(reinterpret_cast<Parament_Context<cuDoubleComplex>*>(handle));
}

Parament_ErrorCode Parament_setHamiltonian_fp64(Parament_Context_f64 *handle, cuDoubleComplex *H0, cuDoubleComplex *H1,
        unsigned int dim, unsigned int amps, bool use_magnus, Parament_QuadratureSpec quadrature_mode) {
    return Parament_setHamiltonian<cuDoubleComplex>(reinterpret_cast<Parament_Context<cuDoubleComplex>*>(handle), H0, H1, dim, amps, use_magnus, quadrature_mode);
}

Parament_ErrorCode Parament_equiprop_fp64(Parament_Context_f64 *handle, cuDoubleComplex *carr, double dt, unsigned int pts,
        unsigned int amps, cuDoubleComplex *out) {
    return Parament_equiprop<cuDoubleComplex>(reinterpret_cast<Parament_Context<cuDoubleComplex>*>(handle), carr, dt, pts, amps, out);
}

Parament_ErrorCode Parament_setIterationCyclesManually_fp64(Parament_Context_f64 *handle, unsigned int cycles) {
    return Parament_setIterationCyclesManually<cuDoubleComplex>(reinterpret_cast<Parament_Context<cuDoubleComplex>*>(handle),
        cycles);
}

Parament_ErrorCode Parament_automaticIterationCycles_fp64(Parament_Context_f64 *handle) {
    return Parament_automaticIterationCycles<cuDoubleComplex>(reinterpret_cast<Parament_Context<cuDoubleComplex>*>(handle));
}

Parament_ErrorCode Parament_peekAtLastError_fp64(Parament_Context_f64 *handle) {
    return Parament_peekAtLastError<cuDoubleComplex>(reinterpret_cast<Parament_Context<cuDoubleComplex>*>(handle));
}