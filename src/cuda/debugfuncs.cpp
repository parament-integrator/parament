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
#include <stdlib.h>
#include <iostream>

#define NO_CUDA_STUBS

#include "debugfuncs.h"

template<typename complex_t>
void readback(complex_t *test, unsigned int dim){
    int len = dim * sizeof(complex_t);
    complex_t* hostprobe = (complex_t *)malloc(len);
    printf("--------------\n");

    printf("Readback of 0x%p\n", test);
    
    cudaMemcpy(hostprobe, test, dim * sizeof(complex_t), cudaMemcpyDeviceToHost);
    printf("Array\n");
    printcomplex(hostprobe, dim);
    printf("--------------\n");
    
    free(hostprobe);
}