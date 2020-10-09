#include <stdlib.h>
#include <assert.h>
#include "parament_context.h"
#include "diagonal_add.h"
#include "mathhelper.h"

#define ENDLINE "\r\n"

#ifndef NDEBUG
    #define PARAMENT_DEBUG(...) {\
        printf("%s", "PARAMENT_DEBUG: ");\
        printf(__VA_ARGS__);\
        printf(ENDLINE);\
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
    struct Parament_Context *paramentHandle = malloc(sizeof(struct Parament_Context));
    *handle_p = paramentHandle;
    if (paramentHandle == NULL) {
        return PARAMENT_STATUS_HOST_ALLOC_FAILED;
    }

    paramentHandle->initialized = false;

    paramentHandle->H0 = NULL;
    paramentHandle->H1 = NULL;
    paramentHandle->one_GPU_diag = NULL;
    paramentHandle->c0 = NULL;
    paramentHandle->c1 = NULL;
    paramentHandle->X = NULL;

    // initialize options
    paramentHandle->MMAX = 11;

    // BESSEL COEFFICIENTS
    paramentHandle->alpha = -2.0;
    paramentHandle->beta = 2.0;

    // Commonly used constants
    paramentHandle->zero = make_cuComplex(0,0);
    paramentHandle->one = make_cuComplex(1,0);
    paramentHandle->two = make_cuComplex(2,0);
    paramentHandle->mone = make_cuComplex(-1,0);
    paramentHandle->mtwo = make_cuComplex(-2,0);
    return PARAMENT_STATUS_SUCCESS;
}

Parament_ErrorCode Parament_init(struct Parament_Context *handle) {
    if (handle->initialized) {
        handle->lastError = PARAMENT_STATUS_ALREADY_INITIALIZED;
        return handle->lastError;
    }

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
    assert(
        cudaSuccess == cudaMemcpy(handle->one_GPU, &handle->one, sizeof(cuComplex), cudaMemcpyHostToDevice)
    );

    // Bessel coefficients
    handle->J = malloc(sizeof(cuComplex) * handle->MMAX);
    if (handle->J == NULL) {
        handle->lastError = PARAMENT_STATUS_HOST_ALLOC_FAILED;
        goto error_cleanup3;
    }

    J_arr(handle->J, handle->MMAX, 2.0);  // compute coefficients
    handle->curr_max_pts = 0; // No points yet allocated

    if (cudaSuccess != cudaDeviceGetAttribute(&handle->numSMs, cudaDevAttrMultiProcessorCount, 0)) {
        // TODO (Pol): better error code here. also, make the GPU configurable
        handle->lastError = PARAMENT_FAIL;
        goto error_cleanup4;
    }

    handle->initialized = true;

    handle->lastError = PARAMENT_STATUS_SUCCESS;
    return handle->lastError;

error_cleanup4:
    free(handle->J);
error_cleanup3:
    assert(cudaSuccess == cudaFree(handle->one_GPU));
error_cleanup2:
    assert(CUBLAS_STATUS_SUCCESS == cublasDestroy(handle->cublasHandle));
error_cleanup1:
    return handle->lastError;
}

/*
 * Frees a previously allocated hamiltionian. No-op if no hamiltonian has been allocated.
 */
static void freeHamiltonian(struct Parament_Context *handle) {
    assert(cudaSuccess == cudaFree(handle->H0));
    assert(cudaSuccess == cudaFree(handle->H1));
    assert(cudaSuccess == cudaFree(handle->one_GPU_diag));

    handle->H0 = NULL;
    handle->H1 = NULL;
    handle->one_GPU_diag = NULL;
}

/*
 * Frees a previously allocated control field and working memory. No-op if no control field has been allocated.
 */
static void freeWorkingMemory(struct Parament_Context *handle) {
    assert(cudaSuccess == cudaFree(handle->c0));
    assert(cudaSuccess == cudaFree(handle->c1)); 
    assert(cudaSuccess == cudaFree(handle->X));
    assert(cudaSuccess == cudaFree(handle->D0));
    assert(cudaSuccess == cudaFree(handle->D1));
    handle->c0 = NULL;
    handle->c1 = NULL;
    handle->X = NULL;
    handle->D0 = NULL;
    handle->D1 = NULL;
    handle->curr_max_pts = 0;
}

Parament_ErrorCode Parament_destroy(struct Parament_Context *handle) {
    if (NULL == handle)
        return PARAMENT_STATUS_SUCCESS;

    if (handle->initialized) {
        free(handle->J);
        assert(cudaSuccess == cudaFree(handle->one_GPU));
        assert(CUBLAS_STATUS_SUCCESS == cublasDestroy(handle->cublasHandle));
        freeHamiltonian(handle);
        freeWorkingMemory(handle);
    }
    free(handle);
    return PARAMENT_STATUS_SUCCESS;
}

Parament_ErrorCode Parament_setHamiltonian(struct Parament_Context *handle, cuComplex *H0, cuComplex *H1, unsigned int dim) {
    // Hamiltonian might have been set before, deallocate first
    freeHamiltonian(handle);

    handle->dim = dim;

    // Allocate GPU memory
    if (cudaSuccess != cudaMalloc(&handle->H0, dim * dim * sizeof(cuComplex))) {
        handle->lastError = PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        goto error_cleanup;
    }
    if (cudaSuccess != cudaMalloc(&handle->H1, dim * dim * sizeof(cuComplex))) {
        handle->lastError = PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        goto error_cleanup;
    }

    // Transfer to GPU
    assert(cudaSuccess == cudaMemcpy(handle->H0, H0, dim * dim * sizeof(cuComplex), cudaMemcpyHostToDevice));
    assert(cudaSuccess == cudaMemcpy(handle->H1, H1, dim * dim * sizeof(cuComplex), cudaMemcpyHostToDevice));

    // Helper Arrays
    if (cudaSuccess != cudaMalloc(&handle->one_GPU_diag, dim * sizeof(cuComplex))) {
        handle->lastError = PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        goto error_cleanup;
    }
    if (CUBLAS_STATUS_SUCCESS != cublasCaxpy(handle->cublasHandle, dim, &handle->one, handle->one_GPU, 0, handle->one_GPU_diag, 1)) {
        handle->lastError = PARAMENT_STATUS_CUBLAS_FAILED;
        goto error_cleanup;
    }

    // nvtxMarkA("Set Hamiltonian routine completed");
    handle->lastError = PARAMENT_STATUS_SUCCESS;
    return handle->lastError;

error_cleanup:
    freeHamiltonian(handle);
    return handle->lastError;
}

static Parament_ErrorCode equipropTransfer(struct Parament_Context *handle, cuComplex *carr, unsigned int pts) {
    // Allocate memory for c arrays if needed
    if (handle->curr_max_pts < pts) {
        PARAMENT_DEBUG("Need to free c arrays");
        freeWorkingMemory(handle);

        PARAMENT_DEBUG("Need to malloc c arrays");
        unsigned int dim = handle-> dim;
        if (cudaSuccess != cudaMalloc(&handle->c0, pts * sizeof(cuComplex))
                || cudaSuccess != cudaMalloc(&handle->c1, pts * sizeof(cuComplex))
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
    assert(cudaSuccess == cudaMemcpy(handle->c1, carr, pts * sizeof(cuComplex), cudaMemcpyHostToDevice));

    return PARAMENT_STATUS_SUCCESS;
}

static Parament_ErrorCode equipropExpand(struct Parament_Context *handle, unsigned int pts) {
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
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim*dim, pts, 1,
        &handle->one,
        handle->H1, dim*dim,
        handle->c1, 1,
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
    dt = dt*2/(handle->beta - handle->alpha)*2;
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

    // Reduction operation:
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


Parament_ErrorCode Parament_equiprop(struct Parament_Context *handle, cuComplex *carr, float dt, unsigned int pts, cuComplex *out) {
    Parament_ErrorCode error;

    error = equipropTransfer(handle, carr, pts);
    if (PARAMENT_STATUS_SUCCESS != error) {
        handle->lastError = error;
        return handle->lastError;
    }

    error = equipropExpand(handle, pts);
    if (PARAMENT_STATUS_SUCCESS != error) {
        handle->lastError = error;
        return handle->lastError;
    }

    error = equipropPropagate(handle, dt, pts);
    if (PARAMENT_STATUS_SUCCESS != error) {
        handle->lastError = error;
        return handle->lastError;
    }

    error = equipropReduce(handle, pts);
    if (PARAMENT_STATUS_SUCCESS != error) {
        handle->lastError = error;
        return handle->lastError;
    }

    // transfer back
    const unsigned int dim = handle->dim;
    cuComplex *const D1 = handle->D1;
    assert(cudaSuccess == cudaMemcpy(out, D1, dim * dim  * sizeof(cuComplex), cudaMemcpyDeviceToHost));
    
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
        default:
            return "Error";
        // TODO: complete
    }
}
