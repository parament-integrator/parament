import logging
import ctypes
import numpy as np

from .constants import *

import parament.paramentlib

logger = logging.getLogger('Parament')
logging.basicConfig()
logger.setLevel(logging.DEBUG)


class Parament:
    def __init__(self):
        """Foo bar"""
        self._lib = parament.paramentlib.lib
        self._handle = ctypes.c_void_p()
        logger.debug('Created Parament context')
        self._checkError(self._lib.Parament_create(ctypes.byref(self._handle)))
        self.dim = 0

    def destroy(self):
        """Foo bar"""
        logger.debug('Destroying Parament context')
        self._checkError(self._lib.Parament_destroy(self._handle))
        self._handle = None
    
    def __del__(self):
        if self._handle is not None:
            self.destroy()

    def setHamiltonian(self, H0: np.ndarray, H1: np.ndarray, use_magnus=False, quadrature_mode='just propagate'):
        """Foo bar. Add docs here"""
        # todo: input validation...
        if quadrature_mode == 'just propagate':
            modesel = 0
        elif quadrature_mode == 'midpoint':
            modesel = 0x1000000
        elif quadrature_mode == 'simpson':
            modesel = 0x2000000
        else:
            raise ValueError('unkonown quadrature mode selected')
        dim = np.shape(H0)
        dim = dim[0]
        amps = np.shape(H1)
        if len(amps) > 2:
            amps = amps[2]
        else:
            amps = 1

        self.dim = dim
        self.amps = amps
        self._checkError(self._lib.Parament_setHamiltonian(
            self._handle,
            np.complex64(np.asfortranarray(H0)),
            np.complex64(np.asfortranarray(H1)),
            dim, amps, use_magnus, modesel
        ))
        logger.debug("Python setHamiltonian completed")

    def equiprop(self, carr, dt):
        """Foo bar"""
        logger.debug("EQUIPROP PYTHON CALLED")
        output = np.zeros(self.dim**2, dtype=np.complex64, order='F')
        pts = np.shape(carr)[0]
        carr = np.complex64(carr)
        self._checkError(self._lib.Parament_equiprop(self._handle, np.asfortranarray(carr), np.double(dt), np.uint(pts), np.uint(self.amps), output))
        return np.ascontiguousarray(np.reshape(output, (self.dim, self.dim)).T)

    def _getErrorMessage(self, code=None):
        if code is None:
            code = self._lib.Parament_getLastError(self._handle)
        return self._lib.Parament_errorMessage(code).decode()

    def _checkError(self, error_code):
        if error_code == PARAMENT_STATUS_SUCCESS:
            return
        try:
            exceptionClass = {
                PARAMENT_STATUS_HOST_ALLOC_FAILED: MemoryError,
                PARAMENT_STATUS_DEVICE_ALLOC_FAILED: MemoryError,
                PARAMENT_STATUS_CUBLAS_INIT_FAILED: RuntimeError,
                PARAMENT_STATUS_INVALID_VALUE: ValueError,
                PARAMENT_STATUS_CUBLAS_FAILED: RuntimeError,
                PARAMENT_STATUS_SELECT_SMALLER_DT: RuntimeError,
                PARAMENT_STATUS_NO_HAMILTONIAN: RuntimeError,
                PARAMENT_STATUS_INVALID_QUADRATURE_SELECTION: ValueError,
                PARAMENT_FAIL: RuntimeError,
            }[error_code]
        except KeyError as e:
            raise AssertionError('Unknown error code ') from e

        message = self._getErrorMessage(error_code)
        raise exceptionClass(f"Error code {error_code}: {message}")
