#define MMAX 11
#define TEST_DIM 12
#define TEST_PTS 80000

#include <stdio.h>
#include <iostream>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <nvtx3/nvToolsExt.h>
using namespace std;


#include "deviceInfo.cu"
#include "printFuncs.cu"
#include "mathhelper.cu"
#include "errorchecker.cu"

// https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
__global__
void saxpy(cuComplex a, cuComplex *y, int s, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n*s; 
         i += blockDim.x * gridDim.x) 
      {
          // dim*(n-floor(n/dim)) + n for indices
          y[s*(i-i/s)+i] = cuCaddf(a,y[s*(i-i/s)+i]);
      }
}


#include "classDef.cu"

extern "C" __declspec(dllexport) cuComplex test_datatype() {
    return make_cuComplex(1, 2);
}



extern "C" __declspec(dllexport) void f() {
    printf("%s", "Hello world!\n");
    cuComplex test;
    test = test_datatype();
    printf("(%5.3f,%5.3fi) \n", test.x, test.y);
}

int main(int argc, const char* argv[]) {
    cout << "Start of the Test program" << endl; 
    device_info();
       

    string s1 = argc[argv-1];
    if ((s1.find("info") != std::string::npos)) {
        cout << "ABORTED DUE TO INFO FLAG" << endl;
        return 0;
    
    } 

    // if (argv > 2) {}

    // Generate H0 & H1 matrices and c array
    cuComplex* hostH0 = (cuComplex*)malloc(TEST_DIM * TEST_DIM * sizeof(cuComplex));
    cuComplex* hostH1 = (cuComplex*)malloc(TEST_DIM * TEST_DIM * sizeof(cuComplex));
    cuComplex* hostc1  = (cuComplex*)malloc(TEST_PTS * sizeof(cuComplex));
    for (int i = 0; i < (TEST_DIM * TEST_DIM); i++) {
        hostH1[i] = make_cuComplex(i, 0);
    }
    hostH0[0] = make_cuComplex(0,0);
    hostH0[1] = make_cuComplex(1, 0);
    hostH0[2] = make_cuComplex(1, 0);
    hostH0[3] = make_cuComplex(0, 0);
    
    for (int i = 0; i < TEST_PTS; i++) {
        hostc1[i] = make_cuComplex(0, 0);
    }

    
    GPURunner testobj;
    
    testobj.set_hamiltonian(hostH0, hostH1,TEST_DIM);
    cuComplex* outputmat = (cuComplex*)malloc(TEST_DIM * TEST_DIM * sizeof(cuComplex));
    
    for (int i=0;i<3;i++){
        //cout << "Now doing run #";
        //cout << i << endl;
    testobj.equiprop(hostc1, 0.1, TEST_PTS, outputmat);
    }
    //printcomplex(outputmat,TEST_DIM*TEST_DIM);
    cout << "Finished!" <<endl;
    free(hostH0);
    free(hostH1);
    free(hostc1);
    free(outputmat);
    
   //cudaProfilerStop();
    return 0;
}

// compile with nvcc -o hello.dll -lcublas --shared hello2.cpp
// or with nvcc -lcublas -o test.exe hello2.cu
// check with dumpbin /EXPORTS hello.dll