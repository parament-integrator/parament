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


import logging
import ctypes
import numpy as np

from .constants import *

import parament.paramentlib

logger = logging.getLogger('Parament')
logging.basicConfig()
logger.setLevel(logging.CRITICAL)

# Import qutip if available
try:
    import qutip
    qutip_available = True                 
except ImportError:
    qutip_available = False


class Parament:
    """Create a new Parament context.

    The context stores settings and state between function calls. Use :py:meth:`setHamiltonian` to load the Hamiltonians,
    and then call :py:meth:`equiprop` to compute the propagator.

    Parameters
    ----------
    precision : {'fp32', 'fp64'}
        The native precision of the new context.

    Examples
    --------
    >>> gpu_runner = parament.Parament()
    >>> H0 = np.array([[1, 0], [0, -1]])
    >>> H1 = np.array([[0, 1], [1, 0]])
    >>> dt = 1.0
    >>> gpu_runner.setHamiltonian(H0, H1)
    >>> gpu_runner.equiprop(np.zeros(1), dt)
    array([[0.54030234-0.84147096j, 0.        +0.j        ],
       [0.        +0.j        , 0.54030234+0.84147096j]], dtype=complex64)

    See Also
    --------
    :c:func:`Parament_create`: The underlying native function.
    """
    def __init__(self, precision='fp32'):
        if precision not in ('fp32', 'fp64'):
            raise ValueError("precision must be either 'fp32' or 'fp64'")
        self._use_doubles = (precision == 'fp64')

        self._lib = parament.paramentlib.lib
        self._handle = ctypes.c_void_p()
        if self._use_doubles:            
            self._checkError(self._lib.Parament_create_fp64(ctypes.byref(self._handle)))
        else:
            self._checkError(self._lib.Parament_create(ctypes.byref(self._handle)))
        logger.debug('Created Parament context')
        self.dim = 0
        self.amps = 0
        if qutip_available:
            logger.debug("Qutip support enabled")

    def destroy(self):
        """Destroy the context. Will implicitly be called when the object is garbage-collected."""
        logger.debug('Destroying Parament context')
        if self._use_doubles:
            self._checkError(self._lib.Parament_destroy_fp64(self._handle))
        else:
            self._checkError(self._lib.Parament_destroy(self._handle))
        self._handle = None
    
    def __del__(self):
        if self._handle is not None:
            self.destroy()

    def setHamiltonian(self, H0: np.ndarray, *H1, use_magnus=False, quadrature_mode='none'):
        """Load the hamiltonians.

        Parameters
        ----------
        H0 : ndarray
            Drift Hamiltionian.
        *H1 : ndarray
            Interaction Hamiltonians.
        use_magnus : bool
            Enable or disable the 1st order Magnus expansion. When ``true``, pass 'simpson' as `quadrature_mode`.
        quadrature_mode : {'none', 'midpoint', 'simpson'}
            The quadrature rule used for interpolating the control amplitudes.


        See Also
        --------
        :c:func:`Parament_setHamiltonian`: The underlying native function.
        """
        if qutip_available:
            if type(H0) == qutip.Qobj:
                self._use_qutip = True
                self._qutip_envelope = H0
                H0 = H0.data.todense()
            else: 
                self._use_qutip = False
        else:
            self._use_qutip = False
        H0 = np.atleast_2d(H0)
        if quadrature_mode == 'none':
            modesel = PARAMENT_QUADRATURE_NONE
        elif quadrature_mode == 'midpoint':
            modesel = PARAMENT_QUADRATURE_MIDPOINT
        elif quadrature_mode == 'simpson':
            modesel = PARAMENT_QUADRATURE_SIMPSON
        else:
            raise ValueError('unknown quadrature mode selected')
        dim = np.shape(H0)
        dim = dim[0]
        if len(H1) == 0:
            raise ValueError('provide at least 1 control amplitude')
        elif len(H1) == 1:
            if qutip_available:
                if all(isinstance(Hi, qutip.Qobj) for Hi in H1[0]):
                    H1 = [[Hi.data.todense() for Hi in H1[0]]]
            H1 = np.atleast_2d(H1[0])
        else:
            if qutip_available:
                if type(H1[0]) == qutip.Qobj:
                    H1 = [Hi.data.todense() for Hi in H1]
            H1 = np.atleast_2d(H1)
        amps = np.shape(H1)
        if len(amps) > 2:
            amps = amps[0]
            H1 = np.swapaxes(H1,0,2)
        else:
            amps = 1
        self.dim = dim
        self.amps = amps
        if self._use_doubles:
            self._checkError(self._lib.Parament_setHamiltonian_fp64(
                self._handle,
                np.complex128(np.asfortranarray(H0)),
                np.complex128(np.asfortranarray(H1)),
                dim, amps, use_magnus, modesel
            ))
        else:

            self._checkError(self._lib.Parament_setHamiltonian(
                self._handle,
                np.complex64(np.asfortranarray(H0)),
                np.complex64(np.asfortranarray(H1)),
                dim, amps, use_magnus, modesel
            ))
        logger.debug("Python setHamiltonian completed")

    def equiprop(self, carr, dt):
        """Compute the propagator from the Hamiltionians.

        Parameters
        ----------
        carr: ndarrray
            Array of the control field amplitudes.
        dt: float
            Time step.

        Returns
        -------
        ndarray
            The computed propagator

        """
        logger.debug("EQUIPROP PYTHON CALLED")
        pts = np.shape(carr)[0]
        if self._use_doubles:
            output = np.zeros(self.dim**2, dtype=np.complex128, order='F')
            carr = np.complex128(carr)
            self._checkError(self._lib.Parament_equiprop_fp64(self._handle, np.asfortranarray(carr), np.double(dt),
                                                              np.uint(pts), np.uint(self.amps), output))
        else:
            output = np.zeros(self.dim**2, dtype=np.complex64, order='F')
            carr = np.complex64(carr)
            self._checkError(self._lib.Parament_equiprop(self._handle, np.asfortranarray(carr), np.double(dt),
                                                         np.uint(pts), np.uint(self.amps), output))
        outputdata = np.ascontiguousarray(np.reshape(output, (self.dim, self.dim)).T)
        if self._use_qutip:
            outputdata = qutip.Qobj(outputdata,dims=self._qutip_envelope.dims)          
        return outputdata

    def _getErrorMessage(self, code=None):
        """
        Query last status code from C.
        """
        if code is None:
            code = self._lib.Parament_getLastError(self._handle)
        return self._lib.Parament_errorMessage(code).decode()

    def _checkError(self, error_code):
        """Check for C errors and translate to human-readable messages
        """
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

def device_info():
    """Print information about the available CUDA resources to stdout
    """
    _lib = parament.paramentlib.lib
    _lib.device_info()
