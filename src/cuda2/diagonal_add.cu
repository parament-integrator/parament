#include <cuda_runtime_api.h>

#define NO_CUDA_STUBS

#include "diagonal_add.h"

// https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
__global__ void saxpy(cuComplex a, cuComplex *y, int s, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n*s; 
         i += blockDim.x * gridDim.x) 
      {
          // dim*(n-floor(n/dim)) + n for indices
          y[s*(i-i/s)+i] = cuCaddf(a,y[s*(i-i/s)+i]);
      }
}

void diagonal_add(cuComplex num, cuComplex *C_GPU, int batch_size)
{
    saxpy<<<32*numSMs, 256>>>(num, C_GPU, dim, batch_size);
}
