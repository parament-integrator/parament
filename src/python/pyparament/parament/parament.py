import logging
import os
import ctypes
from .errorcodes import *
import numpy as np

logger = logging.getLogger('Parament')
logging.basicConfig()
logger.setLevel(logging.DEBUG)

# Todo: switch Win vs Linux
USE_SHARED_PARAMENT = os.environ.get("USE_SHARED_PARAMENT")

if os.name == "nt":
    if not USE_SHARED_PARAMENT:
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
lib.Parament_setHamiltonian.argtypes = [c_ParamentContext_p, c_cuComplex_p, c_cuComplex_p, ctypes.c_uint]
lib.Parament_equiprop.argtypes = [c_ParamentContext_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_uint, c_cuComplex_p]
#lib.Parament_getLastError.argtypes = [c_ParamentContext_p]
lib.Parament_errorMessage.argtypes = [ctypes.c_int]
lib.Parament_errorMessage.restype = ctypes.c_char_p


class Parament:
    def __init__(self):
        self._handle = ctypes.c_void_p()
        logger.debug('Created Parament context')
        self._checkError(lib.Parament_create(ctypes.byref(self._handle)))

    def destroy(self):
        self._checkError(lib.Parament_destroy(self._handle))
        self._handle = None
    
    def __del__(self):
        if self._handle is not None:
            self.destroy()
    
    def setHamiltonian(self, H0: np.ndarray, H1: np.ndarray):
        # todo: input validation...
        dim = np.shape(H0)
        dim = dim[0]
        self.dim = dim
        self._checkError(lib.Parament_setHamiltonian(
            self._handle,
            np.complex64(np.asfortranarray(H0)),
            np.complex64(np.asfortranarray(H1)),
            dim
        ))

    #def equiprop(self, c: np.ndarray, dt: float):
    #    if not c.dtype == np.

    def _getErrorMessage(self, code=None):
        if code is None:
            code = lib.Parament_getLastError(self._handle)
        return lib.Parament_errorMessage(code).decode()

    def _checkError(self, errorCode):
        if errorCode == PARAMENT_STATUS_SUCCESS:
            return
        try:
            exceptionClass = {
                PARAMENT_STATUS_HOST_ALLOC_FAILED: MemoryError,
                PARAMENT_STATUS_DEVICE_ALLOC_FAILED: MemoryError,
                PARAMENT_STATUS_CUBLAS_INIT_FAILED: RuntimeError,
                PARAMENT_STATUS_INVALID_VALUE: ValueError,
                PARAMENT_STATUS_CUBLAS_FAILED: RuntimeError,
                PARAMENT_STATUS_SELECT_SMALLER_DT: RuntimeError,
                PARAMENT_FAIL: RuntimeError,
            }[errorCode]
        except KeyError as e:
            raise AssertionError('Unknown error code ') from e

        message = self._getErrorMessage(errorCode)
        raise exceptionClass(f"Error code {errorCode}: {message}")
