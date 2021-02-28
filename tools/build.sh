SOURCEFILES=deviceInfo.c diagonal_add.cu main.cpp mathhelper.cpp parament.cpp printFuncs.cpp debugfuncs.cpp control_expansion.cu
NVCCFLAGS=-lcublas --compiler-options -fPIC
OUTPUTDIR=build

mkdir build
cd src/cuda
nvcc $NVCCFLAGS -DPARAMENT_LINK -o ../../$OUTPUTDIR/parament $SOURCEFILES$ || exit $?
nvcc $NVCCFLAGS -DPARAMENT_BUILD_DLL -o ../../$OUTPUTDIR/libparament.so $SOURCEFILES$ || exit $?