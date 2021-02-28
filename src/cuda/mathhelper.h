/* Copyright 2020 Konstantin Herb. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include <cublas_v2.h>

// Build complex datatypes for GPU
template <class complex_t>
complex_t makeComplex(double a, double b);

template <class complex_t>
complex_t calculate_bessel_coeffs(int k, double x);

double OneNorm(cuDoubleComplex* arr, unsigned int dim);


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

// Returns the kth power of the imaginary unit in the cuComplex type
cuComplex imag_power(int k);

// Returns the Bessel function array as cuComplex type
void J_arr(cuComplex* arr, int mmax, double c);

LIBSPEC double OneNorm(cuComplex* arr, unsigned int dim);
LIBSPEC double OneNorm_fp64(cuDoubleComplex* arr, unsigned int dim);

#ifdef __cplusplus
}
#endif