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
from functools import wraps
import numpy as np

from .constants import *

import parament.paramentlib

logger = logging.getLogger('Parament')
logging.basicConfig()
logger.setLevel(logging.CRITICAL)

# Import qutip if available
try:
    import qutip
except ImportError:
    import parament.qutip_mock as qutip
else:
    logger.debug("Qutip support enabled")


def _must_be_alive(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if self._handle is None:
            raise RuntimeError("Attempting to use a context that has been destroyed")
        return f(self, *args, **kwargs)

    return wrapper


class Parament:
    """Create a new Parament context.

    The context stores settings and state between function calls. Use :py:meth:`setHamiltonian` to load the
    Hamiltonians, and then call :py:meth:`equiprop` to compute the propagator.

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
    >>> gpu_runner.set_hamiltonian(H0, H1)
    >>> gpu_runner.equiprop(dt, np.zeros(1))
    array([[0.54030234-0.84147096j, 0.        +0.j        ],
       [0.        +0.j        , 0.54030234+0.84147096j]], dtype=complex64)

    To release the resources allocated by the context, call the :meth:`destroy()` method. This happens automatically if
    the context object is garbage collected by the interpreter. Good practice nevertheless mandates proper and explicit
    lif cycle management.

    If the context is only to be short lived, the recommended way is to use it as a context manager.

    >>> with parament.Parament() as gpu_runner:
    ...     gpu_runner.set_hamiltonian(H0, H1)
    ...     gpu_runner.equiprop(dt, np.zeros(1))
    array([[0.54030234-0.84147096j, 0.        +0.j        ],
       [0.        +0.j        , 0.54030234+0.84147096j]], dtype=complex64)

    The ``with`` statement guarantees that the context is properly destroyed, even in the event of an exception.

    See Also
    --------
    :c:func:`Parament_create`: The underlying core function.
    """
    def __init__(self, precision='fp32'):
        if precision not in ('fp32', 'fp64'):
            raise ValueError("precision must be either 'fp32' or 'fp64'")
        self._use_doubles = (precision == 'fp64')

        self._lib = parament.paramentlib.lib
        self._handle = ctypes.c_void_p()
        if self._use_doubles:            
            self._check_error(self._lib.Parament_create_fp64(ctypes.byref(self._handle)))
        else:
            self._check_error(self._lib.Parament_create(ctypes.byref(self._handle)))
        logger.debug('Created Parament context')
        self.dim = -1
        self.amps = -1
        self._qutip_H0 = None
        self._use_qutip = False

    @_must_be_alive
    def destroy(self):
        """Destroy the context. Will implicitly be called when the object is garbage-collected.

        See Also
        --------
        :c:func:`Parament_destroy`: The underlying core function.
        """
        logger.debug('Destroying Parament context')
        if self._use_doubles:
            self._check_error(self._lib.Parament_destroy_fp64(self._handle))
        else:
            self._check_error(self._lib.Parament_destroy(self._handle))
        self._handle = None
    
    def __del__(self):
        if self._handle is not None:
            self.destroy()

    @_must_be_alive
    def set_hamiltonian(self, H0, *H1, use_magnus=False, quadrature_mode='none'):
        """Load the hamiltonians.

        The drift Hamiltonian `H0` can bei either a numpy `ndarray` or a qutip object. In the latter case, subsequent
        calls to :func:`equiprop` will also return a qutip object.

        The parameter `H1` is variadic, i.e. you can pass multiple matrices:

        >>> Parament.set_hamiltonian(H0, H1, H2)

        If you have a list (or tuple) of the control Hamiltonians, pass them like so:

        >>> H_control = (H1, H2)
        >>> Parament.set_hamiltonian(H0, *H_control)

        This even works if `H_control` is not a tuple or a list, but a single 3D `ndarray`, where the first axis
        enumerates the hamiltonians.

        Note that because `H1` is variadic, any following arguments (`use_use_magnus` and `quadrature_mode`) can only be
        passed by keyword.

        Parameters
        ----------
        H0 : ndarray or qutip object
            Drift Hamiltionian.
        *H1 : ndarray, qutip object or list of either one
            Interaction Hamiltonians.
        use_magnus : bool
            Enable or disable the 1st order Magnus expansion. When ``true``, pass 'simpson' as `quadrature_mode`.
        quadrature_mode : {'none', 'midpoint', 'simpson'}
            The quadrature rule used for interpolating the control amplitudes.

        See Also
        --------
        :c:func:`Parament_setHamiltonian`: The underlying core function.
        """

        if isinstance(H0, qutip.Qobj):
            self._use_qutip = True
            self._qutip_H0 = H0
            H0 = H0.data.todense()
        else:
            self._use_qutip = False
        H0 = np.atleast_2d(H0)

        if quadrature_mode == 'none':
            mode_sel = PARAMENT_QUADRATURE_NONE
        elif quadrature_mode == 'midpoint':
            mode_sel = PARAMENT_QUADRATURE_MIDPOINT
        elif quadrature_mode == 'simpson':
            mode_sel = PARAMENT_QUADRATURE_SIMPSON
        else:
            raise ValueError('unknown quadrature mode selected')

        dim = np.shape(H0)
        dim = dim[0]
        if len(H1) == 0:
            raise ValueError('provide at least 1 control amplitude')
        else:
            # convert qutip objects into dense ndarrays
            H1 = [
                Hi.data.todense() if isinstance(Hi, qutip.Qobj) else Hi
                for Hi in H1
            ]
            H1 = np.atleast_2d(H1)
        amps = np.shape(H1)
        if len(amps) > 2:
            amps = amps[0]
        else:
            amps = 1
        self.dim = dim
        self.amps = amps
        # Set Hamiltonians in C order such that we do not need to flip the coefficient arrays to get the correct time
        # ordering.
        if self._use_doubles:
            self._check_error(self._lib.Parament_setHamiltonian_fp64(
                self._handle,
                np.complex128(np.ravel(H0, order='C')),
                np.complex128(np.ravel(H1, order='C')),
                dim, amps, use_magnus, mode_sel
            ))
        else:
            self._check_error(self._lib.Parament_setHamiltonian(
                self._handle,
                np.complex64(np.ravel(H0, order='C')),
                np.complex64(np.ravel(H1, order='C')),
                dim, amps, use_magnus, mode_sel
            ))
        logger.debug("Python setHamiltonian completed")

    @_must_be_alive
    def equiprop(self, dt, *carr):
        """Compute the propagator from the Hamiltionians.

        The `carr` argument is variadic. In the case of multiple control fields, just pass one vector per control
        Hamoltonian. For example, if `u1` and `u2` are the amplitudes for two Hamiltonians:

        >>> equiprop(dt, u1, u2)

        If instead you have a 2D array (where the first axis indexes the fields, and the second axis denotes time),
        or a list/tuple of vectors, you can conveniently unpack it with the `*` operator. For instance, the above call
        is equivalent to:

        >>> carr = u1, u2
        >>> equiprop(dt, *carr)

        Parameters
        ----------
        dt: float
            Time step.
        *carr: ndarrray
            Array of the control field amplitudes.

        Returns
        -------
        ndarray
            The computed propagator

        See Also
        --------
        :c:func:`Parament_equiprop`: The underlying core function.

        """
        logger.debug("EQUIPROP PYTHON CALLED")

        if self.amps < 0:
            # This error is also caught by paramentlib. We catch it here, otherwise it would get confused for a
            # excessive number of control amplitudes below.
            raise RuntimeError("No hamiltonian set")

        amps = len(carr)
        if len(carr) > self.amps:
            raise ValueError(f'Got {len(carr)} amplitude arrays, but there are only {self.amps} Hamiltonians.')

        pts = np.shape(carr[0])[0]
        if any(np.shape(carri) != (pts,) for carri in carr):
            raise ValueError("All amplitude arrays must have the same length.")

        if self._use_doubles:
            output = np.zeros(self.dim**2, dtype=np.complex128, order='C')
            carr = np.complex128(carr)
            self._check_error(self._lib.Parament_equiprop_fp64(self._handle, np.ravel(carr, order='C'), np.double(dt),
                                                               np.uint(pts), np.uint(amps), output))
        else:
            output = np.zeros(self.dim**2, dtype=np.complex64, order='C')
            carr = np.complex64(carr)
            self._check_error(self._lib.Parament_equiprop(self._handle, np.ravel(carr, order='C'), np.float(dt),
                                                          np.uint(pts), np.uint(amps), output))
        output_data = np.ascontiguousarray(np.reshape(output, (self.dim, self.dim)))
        if self._use_qutip:
            output_data = qutip.Qobj(output_data, dims=self._qutip_H0.dims)
        return output_data

    @_must_be_alive
    def _get_error_message(self, code=None):
        """
        Query last status code from C.
        """
        if code is None:
            code = self._lib.Parament_getLastError(self._handle)
        return self._lib.Parament_errorMessage(code).decode()

    def _check_error(self, error_code):
        """Check for C errors and translate to human-readable messages
        """
        if error_code == PARAMENT_STATUS_SUCCESS:
            return
        try:
            exception_class = {
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

        message = self._get_error_message(error_code)
        raise exception_class(f"Error code {error_code}: {message}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()

