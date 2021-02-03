import os
import pathlib
import ctypes
import numpy as np

# Todo: switch Win vs Linux
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
else:
    raise RuntimeError("Don't know how to load library on Linux")

c_ParamentContext_p = ctypes.c_void_p  # void ptr is a good enough abstraction :)
c_cuComplex_p = np.ctypeslib.ndpointer(np.complex64)

# define argument and return types
lib.Parament_create.argtypes = [ctypes.POINTER(c_ParamentContext_p)]
lib.Parament_destroy.argtypes = [c_ParamentContext_p]
lib.Parament_setHamiltonian.argtypes = [c_ParamentContext_p, c_cuComplex_p, c_cuComplex_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_bool, ctypes.c_int]
lib.Parament_equiprop.argtypes = [c_ParamentContext_p, c_cuComplex_p, ctypes.c_float, ctypes.c_uint, ctypes.c_uint, c_cuComplex_p]
#lib.Parament_getLastError.argtypes = [c_ParamentContext_p]
lib.Parament_errorMessage.argtypes = [ctypes.c_int]
lib.Parament_errorMessage.restype = ctypes.c_char_p
