# Working title: GAME 
**G**PU **A**ccelerated **M**atrix **E**xponentiation

## Compiling
Go to the subfolder ``cuda``

Run 
```
nvcc -o hello.dll -lcublas --shared hello2.cpp
```
to compile the dll (on Windows) or the shared library (on Linux).

Run 
```
nvcc -lcublas -o test.exe hello2.cu
```
to compile the test program.

## Check DLL exports
```
dumpbin /EXPORTS hello.dll
```

## Installation (Windows)
1. Install Microsoft Visual Studio (Community Edition is sufficient)
2. Install CUDA tools
3. Add `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.26.28801\bin\Hostx64\x64` to system path (Environment variables)
4. To use NVVP (Viszual profiler) install Java (64 bit for 64 bit system!)
5. Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\extras\CUPTI\lib64` to the system path
6. Allow standard useres to run profiling: Use the Windows file manager to navigate to C:\Program Files\NVIDIA Corporation\Control Panel Client, then right-click nvcplui.exe and select Run as administrator. If the Developer module is not visible, then click Desktop from the menu bar and check Enable Developer settings. Enable Allow access to the GPU performance counters to all users.





