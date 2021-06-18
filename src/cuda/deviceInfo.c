/* Copyright 2020 Konstantin Herb. All Rights Reserved.

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

#include <cuda_runtime_api.h>
#include <stdio.h>

#if defined(_MSC_VER)
    // Microsoft
    #define LIBSPEC __declspec(dllexport)
#elif defined(__GNUC__)
    // GCC 
    #define LIBSPEC __attribute__((visibility("default")))
#else
    // do nothing and hope for the best?
    #define LIBSPEC
#endif

// Displays the available GPUs in stdout
LIBSPEC void device_info(void) {
    int nDevices;
    cudaError_t error;
    error = cudaGetDeviceCount(&nDevices);
    if(cudaSuccess != error) {
        printf("Failed to query the number of CUDA devices. Error code: %d\n", error);
        return;
    }
    printf("PARAMENT_INFO:\n");
    printf("Total number of CUDA devices: %d\n", nDevices);
    printf("-----------------------------------\n");
    for (int i = 0; i < nDevices; i++) {
        struct cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, i);
            if(cudaSuccess != error) {
            printf("Failed to query device properties. Error code: %d", error);
            return;
        }
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
            2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Total global memory: %zd MB\n\n", prop.totalGlobalMem/1024/1024);   
    }
    return;
}
