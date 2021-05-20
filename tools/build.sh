#!/bin/bash
# Copyright 2021 Konstantin Herb, Pol Welter. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ##########################################################################


SOURCEFILES=("deviceInfo.c" "diagonal_add.cu" "main.cpp" "mathhelper.cpp" "parament.cpp" "printFuncs.cpp" "debugfuncs.cpp" "control_expansion.cu")
NVCCFLAGS=("-lcublas" "--compiler-options" "-fPIC")
OUTPUTDIR=build

mkdir -p build
cd src/cuda
nvcc "${NVCCFLAGS[@]}" -DPARAMENT_LINK -DNDEBUG -o "../../$OUTPUTDIR/parament.bin" "${SOURCEFILES[@]}" || exit $?
nvcc "${NVCCFLAGS[@]}" -DPARAMENT_BUILD_DLL -DNDEBUG -o "../../$OUTPUTDIR/libparament.so" "${SOURCEFILES[@]}" || exit $?

cd ../..
