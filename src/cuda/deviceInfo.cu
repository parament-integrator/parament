extern "C" __declspec(dllexport) void device_info() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    printf("Total number of CUDA devices: %d\n", nDevices);
    printf("--------------------------------\n");
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
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