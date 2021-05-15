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


import os
import pathlib
import ctypes
import numpy as np

USE_SHARED_PARAMENT = os.environ.get("USE_SHARED_PARAMENT")
PARAMENT_LIB_DIR = os.environ.get("PARAMENT_LIB_DIR")

if os.name == "nt":
    if PARAMENT_LIB_DIR:
        lib_path = str(pathlib.Path(PARAMENT_LIB_DIR) / 'parament.dll')
    elif not USE_SHARED_PARAMENT:
        lib_path = os.path.dirname(__file__) + '/parament.dll'
    else:
        lib_path = 'parament.dll'  # just search the system path

    try:
        lib = ctypes.CDLL(lib_path, winmode=0)
    except TypeError:
        # the winmode argument has been introduced in Python 3.8
        lib = ctypes.CDLL(lib_path)

elif os.name == "posix":
    if PARAMENT_LIB_DIR:
        lib_path = str(pathlib.Path(PARAMENT_LIB_DIR) / 'libparament.so')
    elif not USE_SHARED_PARAMENT:
        lib_path = os.path.dirname(__file__) + '/libparament.so'
    else:
        lib_path = 'libparament.so'  # just search the system path

    lib = ctypes.cdll.LoadLibrary(lib_path)

else:
    raise RuntimeError("Don't know how to load library on " + os.name)

c_ParamentContext_p = ctypes.c_void_p  # void ptr is a good enough abstraction :)
c_cuComplex_p = np.ctypeslib.ndpointer(np.complex64)
c_cuDoubleComplex_p = np.ctypeslib.ndpointer(np.complex128)


# define argument and return types
lib.Parament_create.argtypes = [ctypes.POINTER(c_ParamentContext_p)]
lib.Parament_destroy.argtypes = [c_ParamentContext_p]
lib.Parament_setHamiltonian.argtypes = [c_ParamentContext_p, c_cuComplex_p, c_cuComplex_p, ctypes.c_uint, ctypes.c_uint,
                                        ctypes.c_bool, ctypes.c_int]
lib.Parament_equiprop.argtypes = [c_ParamentContext_p, c_cuComplex_p, ctypes.c_double, ctypes.c_uint, ctypes.c_uint, c_cuComplex_p]
#lib.Parament_getLastError.argtypes = [c_ParamentContext_p]
lib.Parament_errorMessage.argtypes = [ctypes.c_int]
lib.Parament_errorMessage.restype = ctypes.c_char_p

lib.Parament_create_fp64.argtypes = [ctypes.POINTER(c_ParamentContext_p)]
lib.Parament_destroy_fp64.argtypes = [c_ParamentContext_p]
lib.Parament_setHamiltonian_fp64.argtypes = [c_ParamentContext_p, c_cuDoubleComplex_p, c_cuDoubleComplex_p,
                                             ctypes.c_uint, ctypes.c_uint, ctypes.c_bool, ctypes.c_int]
lib.Parament_equiprop_fp64.argtypes = [c_ParamentContext_p, c_cuDoubleComplex_p, ctypes.c_double, ctypes.c_uint,
                                       ctypes.c_uint, c_cuDoubleComplex_p]
lib.device_info.argtypes = []
