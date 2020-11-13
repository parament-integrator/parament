#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "parament_context.h"
#include "diagonal_add.h"
#include "mathhelper.h"
#include "debugfuncs.h"

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


Parament_ErrorCode Parament_create(struct Parament_Context **handle_p) {
    struct Parament_Context *handle = malloc(sizeof(struct Parament_Context));
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

    // BESSEL COEFFICIENTS
    handle->alpha = -2.0;
    handle->beta = 2.0;

    // Commonly used constants
    handle->zero = make_cuComplex(0,0);
    handle->one = make_cuComplex(1,0);
    handle->two = make_cuComplex(2,0);
    handle->mone = make_cuComplex(-1,0);
    handle->mtwo = make_cuComplex(-2,0);

    // initialize cublas context
    if (CUBLAS_STATUS_SUCCESS != cublasCreate(&(handle->cublasHandle))) {
        handle->lastError = PARAMENT_STATUS_CUBLAS_INIT_FAILED;
        goto error_cleanup1;
    }

    // initialize device memory
    if (cudaSuccess != cudaMalloc(&handle->one_GPU, sizeof(cuComplex))) {
        handle->lastError = PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        goto error_cleanup2;
    }
    cudaError_t error;
    error = cudaMemcpy(handle->one_GPU, &handle->one, sizeof(cuComplex), cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);

    handle->curr_max_pts = 0; // No points yet allocated

    if (cudaSuccess != cudaDeviceGetAttribute(&handle->numSMs, cudaDevAttrMultiProcessorCount, 0)) {
        // TODO (Pol): better error code here. also, make the GPU configurable
        handle->lastError = PARAMENT_FAIL;
        goto error_cleanup3;
    }
    PARAMENT_DEBUG("created Parament context");
    handle->lastError = PARAMENT_STATUS_SUCCESS;
    return handle->lastError;

error_cleanup3:
    error = cudaFree(handle->one_GPU);
    assert(error == cudaSuccess);
error_cleanup2:
    error = cublasDestroy(handle->cublasHandle);
    assert(error == CUBLAS_STATUS_SUCCESS);
error_cleanup1:
    free(handle);
    *handle_p = NULL;
    return handle->lastError;
}

/*
 * Frees a previously allocated hamiltionian. No-op if no hamiltonian has been allocated.
 */
static void freeHamiltonian(struct Parament_Context *handle) {
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
static void freeWorkingMemory(struct Parament_Context *handle) {
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

Parament_ErrorCode Parament_destroy(struct Parament_Context *handle) {
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

Parament_ErrorCode Parament_setHamiltonian(struct Parament_Context *handle, cuComplex *H0, cuComplex *H1, unsigned int dim, unsigned int amps) {
    // Hamiltonian might have been set before, deallocate first
    freeHamiltonian(handle);

    handle->dim = dim;

    // Allocate GPU memory
    if (cudaSuccess != cudaMalloc(&handle->H0, dim * dim * sizeof(cuComplex))) {
        handle->lastError = PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        goto error_cleanup;
    }
    if (cudaSuccess != cudaMalloc(&handle->H1, dim * dim * amps * sizeof(cuComplex))) {
        handle->lastError = PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        goto error_cleanup;
    }

    // Transfer to GPU
    cudaError_t error;
    error = cudaMemcpy(handle->H0, H0, dim * dim * sizeof(cuComplex), cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);
    error = cudaMemcpy(handle->H1, H1, dim * dim * amps * sizeof(cuComplex), cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);

    // Helper Arrays
    if (cudaSuccess != cudaMalloc(&handle->one_GPU_diag, dim * sizeof(cuComplex))) {
        handle->lastError = PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        goto error_cleanup;
    }
    if (CUBLAS_STATUS_SUCCESS != cublasCaxpy(handle->cublasHandle, dim, &handle->one, handle->one_GPU, 0, handle->one_GPU_diag, 1)) {
        handle->lastError = PARAMENT_STATUS_CUBLAS_FAILED;
        goto error_cleanup;
    }

    PARAMENT_DEBUG("Copied to GPU");

    // Determine norm. We use the 1-norm as an upper limit
    handle->Hnorm = OneNorm(H0,dim);
    handle->alpha = -handle->Hnorm;
    handle->beta = handle->Hnorm;
	
    
    // nvtxMarkA("Set Hamiltonian routine completed");
    handle->lastError = PARAMENT_STATUS_SUCCESS;
    return handle->lastError;

error_cleanup:
    freeHamiltonian(handle);
    return handle->lastError;
}

static Parament_ErrorCode equipropComputeCoefficients(struct Parament_Context *handle, float dt) {
    // If enabled, automatically determine number of iterations
    if (!handle->MMAX_manual){
        int MMAX_selected;
        MMAX_selected = Select_Iteration_cycles_fp32(handle->Hnorm, dt);
        if (MMAX_selected == -1){
            return PARAMENT_STATUS_SELECT_SMALLER_DT;
        }
        
        handle->MMAX = MMAX_selected;
    }

    PARAMENT_DEBUG("MMAX is %d\n",handle->MMAX);

    // Allocate Bessel coefficients
    free(handle->J);
    handle->J = NULL;
    handle->J = malloc(sizeof(cuComplex) * (handle->MMAX+1));
    if (handle->J == NULL) {
        return PARAMENT_STATUS_HOST_ALLOC_FAILED;
    }

    // Compute Bessel coefficients
    float x = dt*(handle->beta - handle->alpha)/2;
    for (int k = 0; k < handle->MMAX + 1; k++) {
        handle->J[k] = cuCmulf(imag_power(k), make_cuComplex(_jn(k, x), 0));
    }
    return PARAMENT_STATUS_SUCCESS;
}

static Parament_ErrorCode equipropTransfer(struct Parament_Context *handle, cuComplex *carr, unsigned int pts, unsigned int amps) {
    // Allocate memory for c arrays if needed
    if (handle->curr_max_pts < pts) {
        PARAMENT_DEBUG("Need to free arrays on GPU");
        freeWorkingMemory(handle);

        PARAMENT_DEBUG("Need to malloc arrays on GPU");
        unsigned int dim = handle-> dim;
        if (cudaSuccess != cudaMalloc(&handle->c0, pts * sizeof(cuComplex))
                || cudaSuccess != cudaMalloc(&handle->c1, pts * amps * sizeof(cuComplex))
                || cudaSuccess != cudaMalloc(&handle->X, dim * dim * pts * sizeof(cuComplex))
                || cudaSuccess != cudaMalloc(&handle->D0, dim * dim * pts * sizeof(cuComplex))
                || cudaSuccess != cudaMalloc(&handle->D1, dim * dim * pts * sizeof(cuComplex))) {
            freeWorkingMemory(handle);
            return PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        }

        // Memorize how many pts are initalized
        handle->curr_max_pts = pts;
        
        // Fill c0 array with ones
        if (CUBLAS_STATUS_SUCCESS != cublasCscal(handle->cublasHandle, pts, &handle->zero, handle->c0, 1)) {
            freeWorkingMemory(handle);
            return PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        }
        if (CUBLAS_STATUS_SUCCESS != cublasCaxpy(handle->cublasHandle, pts, &handle->one, handle->one_GPU, 0, handle->c0, 1)) {
            freeWorkingMemory(handle);
            return PARAMENT_STATUS_CUBLAS_FAILED;
        }
    }

    // Transfer c1
    cudaError_t error = cudaMemcpy(handle->c1, carr, amps*pts * sizeof(cuComplex), cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);

    return PARAMENT_STATUS_SUCCESS;
}

static Parament_ErrorCode equipropExpand(struct Parament_Context *handle, unsigned int pts, unsigned int amps) {
    unsigned int dim = handle->dim;
    cublasStatus_t error;
    error = cublasCgemm(handle->cublasHandle,
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
    
    error = cublasCgemm(handle->cublasHandle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim*dim, pts, amps,
        &handle->one,
        handle->H1, dim*dim,
        handle->c1, pts,
        &handle->one,
        handle->X, dim*dim);
    if (error != CUBLAS_STATUS_SUCCESS) {
        return PARAMENT_STATUS_CUBLAS_FAILED;
    }

    return PARAMENT_STATUS_SUCCESS;
}

static Parament_ErrorCode equipropPropagate(struct Parament_Context *handle, float dt, unsigned int pts) {
    // define some short-form aliases...
    const unsigned int dim = handle->dim;
    cuComplex *const D0 = handle->D0;
    cuComplex *const D1 = handle->D1;
    cuComplex *const X = handle->X;

    cublasStatus_t error;

    // Rescale dt
    dt = 2/(handle->beta - handle->alpha)*2;
    cuComplex dt_complex = make_cuComplex(dt, 0);

    cuComplex* ptr_accumulate;

    for (int k = handle->MMAX; k >= 0; k--) {
        if (k == handle->MMAX){
            error = cublasCscal(handle->cublasHandle, pts*dim*dim, &handle->zero, D0, 1);
            if (error != CUBLAS_STATUS_SUCCESS)
                return PARAMENT_STATUS_CUBLAS_FAILED;
        }
        else {
            // D0 = D0 + 2 X @ D1 * dt
            error = cublasCgemmStridedBatched(handle->cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_T,
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
        cudaDeviceSynchronize();
        assert(cudaPeekAtLastError() == cudaSuccess);
        
        // Next step
        k--;

        if (k == handle->MMAX - 1) {
            ptr_accumulate = &handle->zero;
            //cublasCscal(handle, pts*dim*dim, &zero, D1, 1);
        }         
        if (k == 0){
            ptr_accumulate = &handle->mtwo;
        }

        // D1 = D1 + 2 X @ D0
        error = cublasCgemmStridedBatched(handle->cublasHandle,
            CUBLAS_OP_N, CUBLAS_OP_T,
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
       cudaDeviceSynchronize();
       assert(cudaPeekAtLastError() == cudaSuccess);

       if (k == handle->MMAX - 1){
           ptr_accumulate = &handle->mone;
       }
    } 
    // D1 contains now the matrix exponentials
    return PARAMENT_STATUS_SUCCESS;
}

static Parament_ErrorCode equipropReduce(struct Parament_Context *handle, unsigned int pts) {
    // define some short-form aliases...
    const unsigned int dim = handle->dim;
    cuComplex *const D1 = handle->D1;

    cublasStatus_t error;

    int remain_pts = pts;
    int pad = 0;
    while (remain_pts > 1){
        pad = remain_pts % 2;
        remain_pts = remain_pts/2;

        error = cublasCgemmStridedBatched(handle->cublasHandle,
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
            error = cublasCcopy(handle->cublasHandle, 
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

int Select_Iteration_cycles_fp32(float H_norm, float dt) {
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

Parament_ErrorCode Parament_setIterationCyclesManually(struct Parament_Context *handle, unsigned int cycles) {
    handle->MMAX = cycles;
    handle->MMAX_manual = true;
    return PARAMENT_STATUS_SUCCESS;
}

Parament_ErrorCode Parament_automaticIterationCycles(struct Parament_Context *handle) {
    handle->MMAX = 11;
    handle->MMAX_manual = false;
    return PARAMENT_STATUS_SUCCESS;
}

Parament_ErrorCode Parament_equiprop(struct Parament_Context *handle, cuComplex *carr, float dt, unsigned int pts, unsigned int amps, cuComplex *out) {
    PARAMENT_DEBUG("Equiprop C called");
    if (handle->H0 == NULL) {
        handle->lastError = PARAMENT_STATUS_NO_HAMILTONIAN;
        return handle->lastError;
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

    handle->lastError = equipropExpand(handle, pts, amps);
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
    cuComplex *const D1 = handle->D1;
    cudaError_t cudaError = cudaMemcpy(out, D1, dim * dim  * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    assert(cudaError == cudaSuccess);
    
    handle->lastError = PARAMENT_STATUS_SUCCESS;
    return PARAMENT_STATUS_SUCCESS;
}

Parament_ErrorCode Parament_peekAtLastError(struct Parament_Context *handle) {
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
    default:
        return "Unknown error code";
    }
}
