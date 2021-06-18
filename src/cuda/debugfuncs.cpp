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
#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>
#include "printFuncs.hpp"


#include "debugfuncs.hpp"

void readback(cuComplex *test, int dim){
    int len = dim * sizeof(cuComplex);
    cuComplex* hostprobe = (cuComplex *)malloc(len);
    printf("--------------\n");

    printf("Readback of 0x%p\n", test);
     
    cudaMemcpy(hostprobe, test, dim * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    
    printf("Array\n");
    printcomplex(hostprobe, dim);
    printf("--------------\n");
    
}

void readback(cuDoubleComplex *test, int dim){
    int len = dim * sizeof(cuDoubleComplex);
    cuDoubleComplex* hostprobe = (cuDoubleComplex *)malloc(len);
    printf("--------------\n");

    printf("Readback of 0x%p\n", test);
    
    cudaMemcpy(hostprobe, test, dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    printf("Array\n");
    printcomplex(hostprobe, dim);
    printf("--------------\n");
    
    free(hostprobe);
    
}