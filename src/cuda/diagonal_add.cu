#include <cublas_v2.h>

#include "diagonal_add.h"

// https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
__global__ void Caxpy_batched(cuComplex a, cuComplex *y, int s, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n*s; 
         i += blockDim.x * gridDim.x) 
      {
          // dim*(n-floor(n/dim)) + n for indices
          y[s*(i-i/s)+i] = cuCaddf(a,y[s*(i-i/s)+i]);
      }
}

__global__ void Zaxpy_batched(cuDoubleComplex a, cuDoubleComplex *y, int s, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n*s; 
         i += blockDim.x * gridDim.x) 
      {
          // dim*(n-floor(n/dim)) + n for indices
          y[s*(i-i/s)+i] = cuCadd(a,y[s*(i-i/s)+i]);
      }
}

void diagonal_add(cuComplex num, cuComplex *C_GPU, int batch_size, unsigned int numSMs, unsigned int dim)
{
    Caxpy_batched<<<32*numSMs, 256>>>(num, C_GPU, dim, batch_size);
}

void diagonal_add(cuDoubleComplex num, cuDoubleComplex *C_GPU, int batch_size, unsigned int numSMs, unsigned int dim)
{
    Zaxpy_batched<<<32*numSMs, 256>>>(num, C_GPU, dim, batch_size);
}
