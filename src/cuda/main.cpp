#include <iostream>
#include <assert.h>
#include "deviceinfo.h"
#include <cublas_v2.h>
#define NO_CUDA_STUBS
#include "parament.h"
#include "debugfuncs.h"

using namespace std;

#define TEST_DIM 2
#define TEST_PTS 2


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
    hostH0[2] = make_cuComplex(-1, 0);
    hostH0[3] = make_cuComplex(0, 0);
    
    for (int i = 0; i < TEST_PTS; i++) {
        hostc1[i] = make_cuComplex(0, 0);
    }

    Parament_Context* parament;
    Parament_ErrorCode error;
    error = Parament_create(&parament);
    assert(error == PARAMENT_STATUS_SUCCESS);
    error = Parament_setHamiltonian(parament, hostH0, hostH1, TEST_DIM, 1);
    assert(error == PARAMENT_STATUS_SUCCESS);
    
    cuComplex* outputmat = (cuComplex*)malloc(TEST_DIM * TEST_DIM * sizeof(cuComplex));
    
    for (int i=0;i<3;i++){
        cout << "Now doing run #";
        cout << i << endl;
        Parament_equiprop(parament, hostc1, 0.1, TEST_PTS, 1, outputmat);
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