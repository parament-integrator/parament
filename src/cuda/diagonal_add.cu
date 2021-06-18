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
