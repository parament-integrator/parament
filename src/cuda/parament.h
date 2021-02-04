#ifndef PARAMENT_H_
#define PARAMENT_H_

#ifdef __cplusplus
extern "C" {
#else
#include<stdbool.h>
#endif

#ifdef PARAMENT_BUILD_DLL
    #define LIBSPEC __declspec(dllexport)
#elif defined(PARAMENT_LINK)
    #define LIBSPEC
#else
    #define LIBSPEC __declspec(dllimport)
#endif

#ifndef NO_CUDA_STUBS
    typedef struct cuComplex cuComplex;
    typedef struct cuDoubleComplex cuDoubleComplex;
#endif  // NO_CUDA_STUBS

struct Parament_Context_f32;

/**
 * Quadrature rule used by :c:func:`Parament_setHamiltonian`.
 */
typedef enum Parament_QuadratureSpec
{
    PARAMENT_QUADRATURE_NONE = 0x00000000,  /**<No quadrature rule.*/
    PARAMENT_QUADRATURE_MIDPOINT = 0x01000000,  /**<Apply midpoint rule.*/
    PARAMENT_QUADRATURE_SIMPSON = 0x02000000  /**<Apply Simpson's rule.*/
} Parament_QuadratureSpec;

/**
 * Error codes returned by Parament.
 */
typedef enum Parament_ErrorCode {
    /**
     * No error
     */
    PARAMENT_STATUS_SUCCESS = 0,

    /**
     * Memory allocation on the host failed. This is usually a failure of ``malloc()``.
     * 
     * The machine is likely running out of memory. Deallocate memory that is no longer needed.
     */
    PARAMENT_STATUS_HOST_ALLOC_FAILED = 10,

    /**
     * Memory allocation on the device failed. This error usually arises from a failure of ``cudaMalloc()``.
     * 
     * Deallocate memory that is no longer needed.
     */
    PARAMENT_STATUS_DEVICE_ALLOC_FAILED = 20,

    /**
     * Failed to initialize the cuBLAS library.
     */
    PARAMENT_STATUS_CUBLAS_INIT_FAILED = 30,

    /**
     * Trying to call a Parament function with an illegal value.
     */
    PARAMENT_STATUS_INVALID_VALUE = 50,

    /**
     * Failed to execute cuBLAS function.
     */
    PARAMENT_STATUS_CUBLAS_FAILED = 60,

    /**
     * Failed to perform automatic iteration cycles determination.
     */
    PARAMENT_STATUS_SELECT_SMALLER_DT = 70,

    /**
     * Trying to call :c:func:`Parament_equiprop` without having loaded a Hamiltonian via
     * :c:func:`Parament_setHamiltonian`.
     */
    PARAMENT_STATUS_NO_HAMILTONIAN = 80,

    /**
     * Failed to perform automatic iteration cycles determination.
     */
    PARAMENT_STATUS_INVALID_QUADRATURE_SELECTION = 90,

    /**
     * Place holder for more error codes...
     */
    PARAMENT_FAIL = 1000
    // ...
} Parament_ErrorCode;

/**
 * Create a new Parament context.
 * 
 * The context must eventually be destroyed by calling :c:func:`Parament_destroy`.
 *
 * Parameters
 * ----------
 * handle_p:
 *     The new context.
 *
 * Returns
 * -------
 * :c:enum:`Parament_ErrorCode`
 *     - :c:enumerator:`PARAMENT_STATUS_SUCCESS` on success.
 *     - :c:enumerator:`PARAMENT_STATUS_HOST_ALLOC_FAILED` when an underlying call to :c:func:`malloc()` failed.
 *     - :c:enumerator:`PARAMENT_STATUS_CUBLAS_INIT_FAILED` when Parament fails to initialize a cuBLAS context.
 *     - :c:enumerator:`PARAMENT_STATUS_DEVICE_ALLOC_FAILED` when a call to :c:func:`cudaMalloc()` failed.
 *     - :c:enumerator:`PARAMENT_FAIL` when an unknown error occurred.
 */
LIBSPEC Parament_ErrorCode Parament_create(struct Parament_Context_f32 **handle_p);

/**
 * Destroy the context previously created with :c:func:`Parament_create`.
 * 
 * If the provided handle is `NULL`, this function does nothing.
 * Using a context after it has been destroyed results in undefined behaviour.
 *
 * Parameters
 * ----------
 * handle
 *     The context to destroy.
 *
 * Returns
 * -------
 * :c:enum:`Parament_ErrorCode`
 *     - :c:enumerator:`PARAMENT_STATUS_SUCCESS` on success.
 */
LIBSPEC Parament_ErrorCode Parament_destroy(struct Parament_Context_f32 *handle);

/**
 * Load a Hamiltonian.
 *
 * Parameters
 * ----------
 * handle
 *     Handle to the Parament context.
 * H0
 *     Drift Hamiltionian. Must be `dim` x `dim` array.
 * H1
 *     Interaction Hamiltonians. Must be `dim` x `dim` x `amps` array.
 * dim
 *     Dimension of the Hamiltonians.
 * amps
 *     Number of the control amplitudes.
 * use_magnus
 *     Enable or disable the 1st order Magnus expansion. When ``true``, pass
 *     :c:enumerator:`PARAMENT_QUADRATURE_SIMPSON` as `quadrature_mode`.
 * quadrature_mode
 *     The quadrature rule used for interpolating the control amplitudes.
 *
 *       - :c:enumerator:`PARAMENT_QUADRATURE_NONE`
 *       - :c:enumerator:`PARAMENT_QUADRATURE_MIDPOINT`
 *       - :c:enumerator:`PARAMENT_QUADRATURE_SIMPSON`
 *
 * Returns
 * -------
 * :c:enum:`Parament_ErrorCode`
 *     - :c:enumerator:`PARAMENT_STATUS_SUCCESS` on success.
 *     - :c:enumerator:`PARAMENT_STATUS_DEVICE_ALLOC_FAILED` when allocation of memory on the accelerator device failed.
 *     - :c:enumerator:`PARAMENT_STATUS_CUBLAS_FAILED` when an underlying cuBLAS operation failed.
 *     - :c:enumerator:`PARAMENT_STATUS_INVALID_QUADRATURE_SELECTION` when passing `use_magnus=True` without also
 *       passing :c:enumerator:`PARAMENT_QUADRATURE_SIMPSON` as `quadrature_mode`.
 */
LIBSPEC Parament_ErrorCode Parament_setHamiltonian(struct Parament_Context_f32 *handle, cuComplex *H0, cuComplex *H1,
    unsigned int dim, unsigned int amps, bool use_magnus, enum Parament_QuadratureSpec quadrature_mode);

/**
 * Compute the propagator from the Hamiltionians.
 *
 * Parameters
 * ----------
 * context
 *     Handle to the Parament context.
 * carr
 *     Array of the control field amplitudes.
 * dt
 *     Time step.
 * pts
 *     Number of entries per single control vector in `carr`.
 * amps
 *     Number of control Hamiltonians
 * out
 *     The returned propagator.
 * Returns
 * -------
 * :c:enum:`Parament_ErrorCode`
 *     - :c:enumerator:`PARAMENT_STATUS_SUCCESS` on success.
 *     - :c:enumerator:`PARAMENT_STATUS_NO_HAMILTONIAN` when no Hamiltonian has been loaded (see :c:func:`Parament_setHamiltonian`).
 *     - :c:enumerator:`PARAMENT_STATUS_SELECT_SMALLER_DT` when automatic iteration count is enabled, and convergence would require an excessive number of iterations. Reduce the time step `dt`, or see XXXXX.
 *     - :c:enumerator:`PARAMENT_STATUS_DEVICE_ALLOC_FAILED` when allocation of memory on the accelerator device failed.
 *     - :c:enumerator:`PARAMENT_STATUS_CUBLAS_FAILED` when an underlying cuBLAS operation failed.
 */
 LIBSPEC Parament_ErrorCode Parament_equiprop(struct Parament_Context_f32 *handle, cuComplex *carr, float dt, unsigned int pts, unsigned int amps, cuComplex *out);


/**
 * Get the number of Chebychev cycles necessary for reaching machine precision.
 *
 * The iteration count depends on the the given Hamiltonian and the given evolution time.
 *
 * Parameters
 * ----------
 * H_norm
 *     Operator norm of the Hamiltonian.
 * dt
 *     Time step.
 * Returns
 * -------
 * int
 *     The required iteration count. Returns -1 if the product of norm and dt is too large, and convergence cannot be
 *     guaranteed within a reasonable iteration count.
 */
LIBSPEC int Parament_selectIterationCycles_fp32(float H_norm, float dt);

/**
 * Manually enforce the number of iteration cycles used for the Chebychev approximation.
 *
 * Parameters
 * ----------
 * handle
 *     Handle to the Parament context.
 * cycles
 *     Number of iteration cycles.
 *
 * Returns
 * -------
 * :c:enum:`Parament_ErrorCode`
 *     - :c:enumerator:`PARAMENT_STATUS_SUCCESS` on success.
 * 
 * See Also
 * --------
 * :c:func:`Parament_automaticIterationCycles` : Restore the default behaviour.
 */
LIBSPEC Parament_ErrorCode Parament_setIterationCyclesManually(struct Parament_Context_f32 *handle, unsigned int cycles);

/**
 * Reenable the automatic choice of the number of iteration cycles used for the Chebychev approximation.
 *  
 * Parameters
 * ----------
 * handle
 *     Handle to the Parament context.
 *
 * Returns
 * -------
 * :c:enum:`Parament_ErrorCode`
 *     - :c:enumerator:`PARAMENT_STATUS_SUCCESS` on success.
 */
LIBSPEC Parament_ErrorCode Parament_automaticIterationCycles(struct Parament_Context_f32 *handle);

/**
 * Query the last error code.
 * 
 * Parameters
 * ----------
 * handle
 *     Handle to the Parament context.
 *
 * Returns
 * -------
 * :c:enum:`Parament_ErrorCode`
 *     The error code returned by the last Parament call. :c:func:`Parament_peekAtLastError` itself does not overwrite
 *     the error code.
 */
LIBSPEC Parament_ErrorCode Parament_peekAtLastError(struct Parament_Context_f32 *handle);

/**
 * Get human readable message from error code.
 *
 * Parameters
 * ----------
 * errorCode
 *     Error code for which to retrieve a message.
 */
LIBSPEC const char *Parament_errorMessage(Parament_ErrorCode errorCode);

#ifdef __cplusplus
}
#endif

#endif  // PARAMENT_H_
