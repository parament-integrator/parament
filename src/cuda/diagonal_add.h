#ifndef DIAGONAL_ADD_H_
#define DIAGONAL_ADD_H_

void diagonal_add(cuComplex num, cuComplex *C_GPU, int batch_size, unsigned int numSMs, unsigned int dim);
void diagonal_add(cuDoubleComplex num, cuDoubleComplex *C_GPU, int batch_size, unsigned int numSMs, unsigned int dim);

#endif // DIAGONAL_ADD_H_