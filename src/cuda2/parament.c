#include <stdlib.h>
#include "parament_context.h"
#include "diagonal_add.h"
#include "mathhelper.h"


// Commonly used constants
#define zero make_cuComplex(0,0)
#define one make_cuComplex(1,0)
#define two make_cuComplex(2,0)
#define mone make_cuComplex(-1,0)
#define mtwo make_cuComplex(-2,0)

#define ENDLINE "\r\n"

#ifdef DEBUG
    #define PARAMENT_ASSERT(condition, message) {\
        if (!condition) {\
            fprintf(stderr, "Parament assertion failed (%s:%d): %s", __FILE__, __LINE__, message);\
            fprintf(stderr, ENDLINE);\
            exit(1);\
        }\
    }
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
    #define PARAMENT_ASSERT(condition, message) do {} while(0)
    #define PARAMENT_DEBUG(...) do {} while(0)
    #define PARAMENT_ERROR(...) do {} while(0)
#endif


Parament_ErrorCode Parament_create(struct Parament_Context **handle_p) {
    struct Parament_Context *paramentHandle = (struct Parament_Context *) malloc(sizeof(struct Parament_Context));
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
    PARAMENT_ASSERT(
        cudaSuccess == cudaMemcpy(handle->one_GPU, &one, sizeof(cuComplex), cudaMemcpyHostToDevice),
        "Failure in cudaMemcpy"
    );

    // Bessel coefficients
    handle->J = (cuComplex *) malloc(sizeof(cuComplex) * paramentHandle->MMAX);
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
    PARAMENT_ASSERT(cudaSuccess == cudaFree(handle->one_GPU), "");
error_cleanup2:
    PARAMENT_ASSERT(CUBLAS_STATUS_SUCCESS == cublasDestroy(handle->cublasHandle), "");
error_cleanup1:
    return handle->lastError;
}

/*
 * Frees a previously allocated hamiltionian. No-op if no hamiltonian has been allocated.
 */
static void freeHamiltonian(struct Parament_Context *handle) {
    PARAMENT_ASSERT(cudaSuccess == cudaFree(handle->H0), "");
    PARAMENT_ASSERT(cudaSuccess == cudaFree(handle->H1), "");
    PARAMENT_ASSERT(cudaSuccess == cudaFree(handle->one_GPU_diag), "");

    handle->H0 = NULL;
    handle->H1 = NULL;
    handle->one_GPU_diag = NULL;
}

/*
 * Frees a previously allocated control field and working memory. No-op if no control field has been allocated.
 */
static void freeWorkingMemory(struct Parament_Context *handle) {
    PARAMENT_ASSERT(cudaSuccess == cudaFree(handle->c0), "");
    PARAMENT_ASSERT(cudaSuccess == cudaFree(handle->c1), ""); 
    PARAMENT_ASSERT(cudaSuccess == cudaFree(handle->X), "");
    PARAMENT_ASSERT(cudaSuccess == cudaFree(handle->D0), "");
    PARAMENT_ASSERT(cudaSuccess == cudaFree(handle->D1), "");
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
        PARAMENT_ASSERT(cudaSuccess == cudaFree(handle->one_GPU), "");
        PARAMENT_ASSERT(CUBLAS_STATUS_SUCCESS == cublasDestroy(handle->cublasHandle), "");
        freeHamiltonian(handle);
        freeWorkingMemory(handle);
    }
    free(handle);
}

Parament_ErrorCode Parament_setHamiltonian(struct Parament_Context *handle, cuComplex *H0, cuComplex *H1, unsigned int dim) {
    // Hamiltonian might have been set before, deallocate first
    freeHamiltonian(handle);

    handle->dim = dim;

    // Allocate GPU memory
    if (cudaSuccess != cudaMalloc(&handle->H0, dim * dim * sizeof(cuComplex)) {
        handle->lastError = PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        goto error_cleanup;
    }
    if (cudaSuccess != cudaMalloc(&handle->H1, dim * dim * sizeof(cuComplex)) {
        handle->lastError = PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        goto error_cleanup;
    }

    // Transfer to GPU
    PARAMENT_ASSERT(cudaSuccess == cudaMemcpy(handle->H0, H0, dim * dim * sizeof(cuComplex), cudaMemcpyHostToDevice));
    PARAMENT_ASSERT(cudaSuccess == cudaMemcpy(handle->H1, H1, dim * dim * sizeof(cuComplex), cudaMemcpyHostToDevice));

    // Helper Arrays
    if (cudaSuccess != cudaMalloc(&handle->one_GPU_diag, dim * sizeof(cuComplex))) {
        handle->lastError = PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        goto error_cleanup;
    }
    if (CUBLAS_STATUS_SUCCESS != cublasCaxpy(handle->cublasHandle, dim, &one, handle->one_GPU, 0, handle->one_GPU_diag, 1)) {
        handle->lastError = PARAMENT_STATUS_CUBLAS_FAILED;
        goto error_cleanup;
    }

    // nvtxMarkA("Set Hamiltonian routine completed");
    handler->lastError = PARAMENT_STATUS_SUCCESS;
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
        if (cudaSuccess != cudaMalloc(&c0, pts * sizeof(cuComplex))
                || cudaSuccess != cudaMalloc(&c1, pts * sizeof(cuComplex))
                || cudaSuccess != cudaMalloc(&X, dim * dim * pts * sizeof(cuComplex))
                || cudaSuccess != cudaMalloc(&D0, dim * dim * pts * sizeof(cuComplex))
                || cudaSuccess != cudaMalloc(&D1, dim * dim * pts * sizeof(cuComplex))) {
            freeWorkingMemory(handle);
            return PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        }

        // Memorize how many pts are initalized
        handle->curr_max_pts = pts;
        
        // Fill c0 array with ones
        if (CUBLAS_STATUS_SUCCESS != cublasCscal(handle, pts, &zero, handle->c0, 1)) {
            freeWorkingMemory(handle);
            return PARAMENT_STATUS_DEVICE_ALLOC_FAILED;
        }
        if (CUBLAS_STATUS_SUCCESS != cublasCaxpy(handle->cublasHandle, pts, &one, handle->one_GPU, 0, handle->c0, 1)) {
            freeWorkingMemory(handle);
            return PARAMENT_STATUS_CUBLAS_FAILED;
        }
    }

    // Transfer c1
    PARAMENT_ASSERT(cudaSuccess == cudaMemcpy(c1, carr, pts * sizeof(cuComplex), cudaMemcpyHostToDevice), "");
}

static Parament_ErrorCode equipropExpand(struct Parament_Context *handle, unsigned int pts) {
    unsigned int dim = handle->dim;
    cublasStatus_t error;
    error = cublasCgemm(handle,
         CUBLAS_OP_N, CUBLAS_OP_N,
         dim*dim, pts, 1,
         &one,
         handle->H0, dim*dim,
         handle->c0, 1,
         &zero,
         handle->X, dim*dim); 
    if (error != CUBLAS_STATUS_SUCCESS) {
        return PARAMENT_STATUS_CUBLAS_FAILED;
    }
    
    error = cublasCgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim*dim, pts, 1,
        &one,
        handle->H1, dim*dim,
        handle->c1, 1,
        &one,
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

    cublas_error_t error;

    // Rescale dt
    dt = dt*2/(beta-alpha)*2;
    cuComplex dt_complex = make_cuComplex(dt, 0);

    cuComplex* ptr_accumulate;

    for (int k = MMAX; k >= 0; k--) {
        if (k == MMAX){
            error = cublasCscal(handle->cublasHandle, pts*dim*dim, &zero, D0, 1);
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
                &mone,
                D0, dim, dim*dim,
                pts
            );
            if (error != CUBLAS_STATUS_SUCCESS)
                return PARAMENT_STATUS_CUBLAS_FAILED;
        }
        
        // D0 = D0 + I*ak
        diagonal_add(J[k], D0, pts);
        
        // Next step
        k--;

        if (k == MMAX-1) {
            ptr_accumulate = &zero;
            //cublasCscal(handle, pts*dim*dim, &zero, D1, 1);
        }         
        if (k == 0){
            ptr_accumulate = &mtwo;
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
       diagonal_add(J[k], D1, pts);

       if (k == MMAX - 1){
           ptr_accumulate = &mone;
       }
    } 
    // D1 contains now the matrix exponentials
    return PARAMENT_STATUS_SUCCESS;
}

static Parament_ErrorCode equipropReduce(struct Parament_Context *handle, unsigned int pts) {
    // define some short-form aliases...
    const unsigned int dim = handle->dim;
    cuComplex *const D1 = handle->D1;

    cublas_error_t error;

    // Reduction operation:
    int remain_pts = pts;
    int pad = 0;
    while (remain_pts > 1){
        pad = remain_pts % 2;
        remain_pts = remain_pts/2;

        error = cublasCgemmStridedBatched(handle->cublasHandle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim, dim, dim,
            &one,
            D1          , dim, dim*dim*2,
            D1 + dim*dim, dim, dim*dim*2,
            &zero,
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
    PARAMENT_ASSERT(cudaSuccess == cudaMemcpy(out, D1, dim * dim  * sizeof(cuComplex), cudaMemcpyDeviceToHost), "");
    
    handle->lastError = PARAMENT_STATUS_SUCCESS;
    return PARAMENT_STATUS_SUCCESS;
}

Parament_ErrorCode Parament_getLastError(struct Parament_Context *handle) {
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
