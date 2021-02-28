#!/bin/bash
SOURCEFILES=("deviceInfo.c" "diagonal_add.cu" "main.cpp" "mathhelper.cpp" "parament.cpp" "printFuncs.cpp" "debugfuncs.cpp" "control_expansion.cu")
NVCCFLAGS=("-lcublas" "--compiler-options" "-fPIC")
OUTPUTDIR=build

mkdir -p build
cd src/cuda
echo nvcc "${NVCCFLAGS[@]}" -DPARAMENT_LINK -o "../../$OUTPUTDIR/parament.bin" "${SOURCEFILES[@]}" || exit $?
nvcc "${NVCCFLAGS[@]}" -DPARAMENT_BUILD_DLL -o "../../$OUTPUTDIR/libparament.so" "${SOURCEFILES[@]}" || exit $?

cd ../..
