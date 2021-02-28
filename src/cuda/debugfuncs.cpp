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