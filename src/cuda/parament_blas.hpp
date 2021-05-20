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


// GEMM Stided batched
cublasStatus_t cublasGgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const cuComplex       *alpha,
    const cuComplex       *A, int lda,
    long long int          strideA,
    const cuComplex       *B, int ldb,
    long long int          strideB,
    const cuComplex       *beta,
    cuComplex             *C, int ldc,
    long long int          strideC,
    int batchCount
) {
    return cublasCgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

cublasStatus_t cublasGgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda,
    long long int          strideA,
    const cuDoubleComplex *B, int ldb,
    long long int          strideB,
    const cuDoubleComplex *beta,
    cuDoubleComplex       *C, int ldc,
    long long int          strideC,
    int batchCount
) {
    return cublasZgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

// GEMM
cublasStatus_t cublasGgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const cuComplex       *alpha,
    const cuComplex       *A, int lda,
    const cuComplex       *B, int ldb,
    const cuComplex       *beta,
    cuComplex       *C, int ldc
) {
   return cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasGgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *B, int ldb,
    const cuDoubleComplex *beta,
    cuDoubleComplex *C, int ldc
) {
    return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// SCAL
cublasStatus_t  cublasGscal(
    cublasHandle_t handle, int n,
    const cuComplex       *alpha,
    cuComplex       *x, int incx
) {
    return cublasCscal(handle, n, alpha, x, incx);
}

cublasStatus_t  cublasGscal(
    cublasHandle_t handle, int n,
    const cuDoubleComplex *alpha,
    cuDoubleComplex *x, int incx
) {
    return cublasZscal(handle, n, alpha, x, incx);
}

// COPY
cublasStatus_t cublasGcopy(
    cublasHandle_t handle, int n,
    const cuComplex       *x, int incx,
    cuComplex             *y, int incy
) {
    return cublasCcopy(handle, n, x, incx, y, incy);
}

cublasStatus_t cublasGcopy(
    cublasHandle_t handle, int n,
    const cuDoubleComplex *x, int incx,
    cuDoubleComplex       *y, int incy
) {
   return cublasZcopy(handle, n, x, incx, y, incy);
}

// AXPY
cublasStatus_t cublasGaxpy(
    cublasHandle_t handle, int n,
    const cuComplex       *alpha,
    const cuComplex       *x, int incx,
    cuComplex             *y, int incy
) {
    return cublasCaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasGaxpy(
    cublasHandle_t handle, int n,
    const cuDoubleComplex *alpha,
    const cuDoubleComplex *x, int incx,
    cuDoubleComplex       *y, int incy
) {
       return cublasZaxpy(handle, n, alpha, x, incx, y, incy);
}

