/* Copyright 2021 Konstantin Herb, Pol Welter. All Rights Reserved.

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


#ifndef CONTROL_EXPANSION_H_
#define CONTROL_EXPANSION_H_


#ifndef NO_CUDA_STUBS
typedef struct cuComplex cuComplex;
#endif  // NO_CUDA_STUBS

void control_magnus(cuComplex* coeff_in, cuComplex *coeff_out, unsigned int amps, unsigned int n, float dt,
                    unsigned int numSMs);
void control_magnus(cuDoubleComplex* coeff_in, cuDoubleComplex *coeff_out, unsigned int amps, unsigned int n,
                    double dt, unsigned int numSMs);


void control_midpoint(cuComplex* coeff_in, cuComplex *coeff_out, unsigned int amps, unsigned int n,
                        unsigned int numSMs);
void control_midpoint(cuDoubleComplex* coeff_in, cuDoubleComplex *coeff_out, unsigned int amps, unsigned int n,
                        unsigned int numSMs);


void control_simpson(cuComplex* coeff_in, cuComplex *coeff_out, unsigned int amps, unsigned int n,
                        unsigned int numSMs);
void control_simpson(cuDoubleComplex* coeff_in, cuDoubleComplex *coeff_out, unsigned int amps, unsigned int n,
                        unsigned int numSMs);


#endif // CONTROL_EXPANSION_H_