#ifndef CONTROL_EXPANSION_H_
#define CONTROL_EXPANSION_H_



#ifndef NO_CUDA_STUBS
typedef struct cuComplex cuComplex;
#endif  // NO_CUDA_STUBS

void control_magnus(cuComplex* coeff_in, cuComplex *coeff_out, unsigned int amps, unsigned int n, float dt, unsigned int numSMs);
void control_magnus(cuDoubleComplex* coeff_in, cuDoubleComplex *coeff_out, unsigned int amps, unsigned int n, float dt, unsigned int numSMs);


void control_midpoint(cuComplex* coeff_in, cuComplex *coeff_out, unsigned int amps, unsigned int n, unsigned int numSMs);
void control_midpoint(cuDoubleComplex* coeff_in, cuDoubleComplex *coeff_out, unsigned int amps, unsigned int n, unsigned int numSMs);


void control_simpson(cuComplex* coeff_in, cuComplex *coeff_out, unsigned int amps, unsigned int n, unsigned int numSMs);
void control_simpson(cuDoubleComplex* coeff_in, cuDoubleComplex *coeff_out, unsigned int amps, unsigned int n, unsigned int numSMs);


#endif // CONTROL_EXPANSION_H_