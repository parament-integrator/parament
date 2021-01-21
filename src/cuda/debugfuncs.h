#ifndef DEBUG_H_
#define DEBUG_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NO_CUDA_STUBS
typedef struct cuComplex cuComplex;
#endif  // NO_CUDA_STUBS


void readback(cuComplex *test, unsigned int dim);

#ifdef __cplusplus
}
#endif

#endif // DEBUG_H_


