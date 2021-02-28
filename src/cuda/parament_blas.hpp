// GEMM Stided batched
cublasStatus_t cublasGgemmStridedBatched(cublasHandle_t handle,
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
                                  int batchCount)
                                  {
                                    return cublasCgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
                                  }

cublasStatus_t cublasGgemmStridedBatched(cublasHandle_t handle,
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
                                  int batchCount)
                                  {
                                    return cublasZgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
                                  }

// GEMM
cublasStatus_t cublasGgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const cuComplex       *alpha,
                           const cuComplex       *A, int lda,
                           const cuComplex       *B, int ldb,
                           const cuComplex       *beta,
                           cuComplex       *C, int ldc)
                           {
                               return cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
                           }

cublasStatus_t cublasGgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *B, int ldb,
                           const cuDoubleComplex *beta,
                           cuDoubleComplex *C, int ldc)
                           {
                               return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
                           }

// SCAL
cublasStatus_t  cublasGscal(cublasHandle_t handle, int n,
                            const cuComplex       *alpha,
                            cuComplex       *x, int incx)
                            {
                                return cublasCscal(handle, n, alpha, x, incx);
                            }

cublasStatus_t  cublasGscal(cublasHandle_t handle, int n,
                            const cuDoubleComplex *alpha,
                            cuDoubleComplex *x, int incx)
                            {
                                return cublasZscal(handle, n, alpha, x, incx);
                            }

// COPY
cublasStatus_t cublasGcopy(cublasHandle_t handle, int n,
                           const cuComplex       *x, int incx,
                           cuComplex             *y, int incy)
                           {
                               return cublasCcopy(handle, n, x, incx, y, incy);
                           }

cublasStatus_t cublasGcopy(cublasHandle_t handle, int n,
                           const cuDoubleComplex *x, int incx,
                           cuDoubleComplex       *y, int incy)
                           {
                               return cublasZcopy(handle, n, x, incx, y, incy);
                           }

// AXPY
cublasStatus_t cublasGaxpy(cublasHandle_t handle, int n,
                           const cuComplex       *alpha,
                           const cuComplex       *x, int incx,
                           cuComplex             *y, int incy)
                           {
                               return cublasCaxpy(handle, n, alpha, x, incx, y, incy);
                           }

cublasStatus_t cublasGaxpy(cublasHandle_t handle, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *x, int incx,
                           cuDoubleComplex       *y, int incy)
                           {
                               return cublasZaxpy(handle, n, alpha, x, incx, y, incy);
                           }

// AXPY Strided batched
// Custom implementation since missing in the CUBLAS library
// WIP: This will replace diagonal_add
