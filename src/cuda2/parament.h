#ifndef PARAMENT_H_
#define PARAMENT_H_

#ifdef __cplusplus
extern "C" {
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
#endif  // NO_CUDA_STUBS

struct Parament_Context;

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
    PARAMENT_STATUS_HOST_ALLOC_FAILED,

    /**
     * Memory allocation on the device failed. This error usually arises from a failure of ``cudaMalloc()``.
     * 
     * Deallocate memory that is no longer needed.
     */
    PARAMENT_STATUS_DEVICE_ALLOC_FAILED,

    /**
     * Failed to initialize the cuBLAS library.
     */
    PARAMENT_STATUS_CUBLAS_INIT_FAILED,

    /**
     * Trying to call :c:func:`Parament_init` with a context that is initialized already.
     */
    PARAMENT_STATUS_ALREADY_INITIALIZED,

    /**
     * Trying to call a Parament function with an illegal value.
     */
    PARAMENT_STATUS_INVALID_VALUE,

    /**
     * Failed to execute cuBLAS function.
     */
    PARAMENT_STATUS_CUBLAS_FAILED,

    /**
     * Failed to beform automatic iteration cycles determination.
     */
    PARAMENT_STATUS_SELECT_SMALLER_DT,

    /**
     * Place holder for more error codes...
     */
    PARAMENT_FAIL = 100
    // ...
} Parament_ErrorCode;

/**
 * Create a new (blank) Parament context.
 * 
 * The context must eventually be destroyed by calling :c:func:`Parament_destroy`.
 *
 * :param handle_p: The new context.
 * :return: 
 *   - :c:enumerator:`PARAMENT_STATUS_SUCCESS` on success.
 *   - :c:enumerator:`PARAMENT_STATUS_HOST_ALLOC_FAILED` when the underlying call to :c:func:`malloc()` failed.
 * 
 */
LIBSPEC Parament_ErrorCode Parament_create(struct Parament_Context **handle_p);

/**
 * Initialize the Parament context.
 * 
 * This function must be called after :c:func:`Parament_create`.
 *
 * :param handle: Handle to the Parament context.
 * :return: 
 *   - :c:enumerator:`PARAMENT_STATUS_SUCCESS` on success.
 *   - :c:enumerator:`PARAMENT_STATUS_ALREADY_INITIALIZED` when calling with an already initialized context.
 *   - :c:enumerator:`PARAMENT_STATUS_CUBLAS_INIT_FAILED` when Parament fails to initialize a cuBLAS context.
 *   - :c:enumerator:`PARAMENT_STATUS_HOST_ALLOC_FAILED` when a call to :c:func:`malloc()` failed.
 *   - :c:enumerator:`PARAMENT_STATUS_DEVICE_ALLOC_FAILED` when a call to :c:func:`cudaMalloc()` failed.
  
 */
LIBSPEC Parament_ErrorCode Parament_init(struct Parament_Context *handle);

/**
 * Destroy the context previously created with :c:func:`Parament_create`.
 * 
 * If the provided handle is `NULL`, this function does nothing.
 *
 * :param handle: The context to destroy.
 * 
 * Using a context after it has been destroyed results in undefined behaviour.
 */
LIBSPEC Parament_ErrorCode Parament_destroy(struct Parament_Context *handle);

/**
 * Load a Hamiltonian.
 * 
 * :param handle: Handle to the Parament context.
 * :param H0: Drift Hamiltionian. Must be `dim` x `dim` array.
 * :param H1: Interaction Hamiltonian. Must be `dim` x `dim` array.
 * :param dim: Dimension of the Hamiltonians.
 */
LIBSPEC Parament_ErrorCode Parament_setHamiltonian(struct Parament_Context *handle, cuComplex *H0, cuComplex *H1, unsigned int dim);

/**
 * Compute the propagator from the Hamiltionian.
 * 
 * :param context: Handle to the Parament context.
 * :param carr: Array of the control field amplitudes.
 * :param dt: Time step.
 * :param pts: Number of entries in `carr`.
 * :param out: The returned propagator.
 */
LIBSPEC Parament_ErrorCode Parament_equiprop(struct Parament_Context *handle, cuComplex *carr, float dt, unsigned int pts, cuComplex *out);


/**
 * Get the number of Chebychev cycles for the given Hamiltonian and the given evolution time that are necessary to reach machine precision.
 * Returns -1 if the product of norm and dt is too large to be handled.
 *  
 * :param H_norm: Operator norm.
 * :param dt: Time step.
 * :param out: Number of iteration cycles.
 */
LIBSPEC int Select_Iteration_cycles_fp32(float H_norm, float dt);

/**
 * Manually enforce the number of iteration cycles used for the Chebychev approximation
 *  
 * :param handle: Handle to the Parament context.
 * :param cycles: Number of iteration cycles.
 */
LIBSPEC Parament_ErrorCode Parament_setIterationCyclesManually(struct Parament_Context *handle, unsigned int cycles);

/**
 * Reenable the automatic choice for the number of iteration cycles used for the Chebychev approximation
 *  
 * :param handle: Handle to the Parament context.
 */
LIBSPEC Parament_ErrorCode Parament_unsetIterationCycles(struct Parament_Context *handle);



/**
 * Query the last error code.
 * 
 * :param handle: Handle to the Parament context.
 */
LIBSPEC Parament_ErrorCode Parament_peekAtLastError(struct Parament_Context *handle);

/**
 * Get human readable message from error code.
 * 
 * :param errorCode: Error code for which to retrieve a message.
 */
LIBSPEC const char *Parament_errorMessage(Parament_ErrorCode errorCode);

#ifdef __cplusplus
}
#endif

#endif  // PARAMENT_H_
