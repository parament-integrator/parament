#ifndef DIAGONAL_ADD_H_
#define DIAGONAL_ADD_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NO_CUDA_STUBS
typedef struct cuComplex cuComplex;
#endif  // NO_CUDA_STUBS

void diagonal_add(cuComplex num, cuComplex *C_GPU, int batch_size, unsigned int numSMs, unsigned int dim);

#ifdef __cplusplus
}
#endif

#endif // DIAGONAL_ADD_H_