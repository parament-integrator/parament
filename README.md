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

## Check DLL expoerts
```
dumpbin /EXPORTS hello.dll
```