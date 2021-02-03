#include <cublas_v2.h>
#include <stdlib.h>

#define NO_CUDA_STUBS

#include "debugfuncs.h"

void readback(cuComplex *test, unsigned int dim){
    int len = dim * sizeof(cuComplex);
    cuComplex* hostprobe = (cuComplex *)malloc(len);
    printf("--------------\n");

    printf("Readback of 0x%p\n", test);
    
    cudaMemcpy(hostprobe, test, dim * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    printf("Array\n");
    printcomplex(hostprobe, dim);
    printf("--------------\n");
    
    free(hostprobe);
}