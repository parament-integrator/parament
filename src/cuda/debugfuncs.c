#include "printFuncs.h"
#include <stdlib.h>

void readback(cuComplex *test, unsigned int dim){

    cuComplex* hostprobe = (cuComplex*)malloc(dim * dim * sizeof(cuComplex));
    cudaMemcpy(hostprobe, test, dim * dim * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    printf("Array\n");
    printcomplex(hostprobe, dim*dim);
    free(hostprobe);
}