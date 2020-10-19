import ctypes
from .errorcodes import *
import numpy as np

# Todo: switch Win vs Linux
lib = ctypes.LoadLibrary('parament.dll')

c_ParamentContext_p = ctypes.c_void_p  # void ptr is a good enough abstraction :)
c_cuComplex_p = ctypes.c_void_p

# define argument and return types
lib.Parament_create.argtypes = [ctypes.POINTER(c_ParamentContext_p)]
lib.Parament_init.argtypes = [c_ParamentContext_p]
lib.Parament_destroy.argtypes = [c_ParamentContext_p]
lib.Parament_setHamiltonian.argtypes = [c_ParamentContext_p, c_cuComplex_p, c_cuComplex_p, ctypes.c_uint]
lib.Parament_equiprop.argtypes = [c_ParamentContext_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_uint, c_cuComplex_p]
lib.Parament_getLastError.argtypes = [c_ParamentContext_p]
lib.Parament_errorMessage.argtypes = [ctypes.c_int]
lib.Parament_errorMessage.restype = c_char_p

class Parament:
    def __init__(self):
        self._handle = ctypes.c_void_p()
        self._checkError(self.lib.Parament_create(ctypes.byref(self._handle)))

    def initialize(self):
        self._checkError(self.lib.Parament_init(self._handle))

    def destroy(self):
        self._checkError(self.lib.Parament_destroy(self._handle))
        self._handle = None
    
    def __del__(self):
        if self._handle is not None:
            self.destroy()
    
    def setHamiltonian(self, H0: np.ndarray, H1: np.ndarray):
        # todo: input validation...
        dim = np.shape(H0)
        dim = dim[0]
        self.dim = dim
        self._checkError(self.lib.Parament_setHamiltonian(self._handle, np.asfortranarray(H0),np.asfortranarray(H1),dim)))

    def equiprop(self, c: np.ndarray, dt: float):
        if not c.dtype == np.

    def _getErrorMessage(self, code=None):
        if code is None:
            code = self.lib.Parament_getLastError(self._handle)
        return self.lib.Parament_errorMessage(code).decode()

    def _checkError(self, errorCode):
        if errorCode == PARAMENT_SUCCESS:
            return
        try:
            exceptionClass = {
                PARAMENT_STATUS_HOST_ALLOC_FAILED: MemoryError,
                PARAMENT_STATUS_DEVICE_ALLOC_FAILED: MemoryError,
                PARAMENT_STATUS_CUBLAS_INIT_FAILED: RuntimeError,
                PARAMENT_STATUS_ALREADY_INITIALIZED: RuntimeError,
                PARAMENT_STATUS_INVALID_VALUE: ValueError,
                PARAMENT_STATUS_CUBLAS_FAILED: RuntimeError,
                PARAMENT_FAIL: RuntimeError,
            }[errorCode]
        except KeyError as e:
            raise AssertionError('Unknown error code') from e

        message = self._getErrorMessage(errorCode)
        raise exceptionClass(errorCode)
