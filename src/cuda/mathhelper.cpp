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

#include <cmath>
#include "mathhelper.h"


template <class complex_t>
complex_t makeComplex(double a, double b){};

template <>
cuComplex makeComplex<cuComplex>(double a, double b) {
    float x = (float) a;
    float y = (float) b;
    return make_cuComplex(x, y);
}

template <> 
cuDoubleComplex makeComplex<cuDoubleComplex>(double a, double b) {
    return make_cuDoubleComplex(a, b);
}


cuComplex imag_power(int k) {
    switch (k % 4) {
        case 0:
            return make_cuComplex(1, 0);
        case 1:
            return make_cuComplex(0, -1);
        case 2:
            return make_cuComplex(-1, 0);
        case 3: default:
            return make_cuComplex(0, 1);
    }
}

cuDoubleComplex imag_power_fp64(int k) {
    switch (k % 4) {
        case 0:
            return make_cuDoubleComplex(1, 0);
        case 1:
            return make_cuDoubleComplex(0, -1);
        case 2:
            return make_cuDoubleComplex(-1, 0);
        case 3: default:
            return make_cuDoubleComplex(0, 1);
    }
}

 inline double BesselJn(int n, double x) { 
 #if defined(CERES_MSVC_USE_UNDERSCORE_PREFIXED_BESSEL_FUNCTIONS) 
   return _jn(n, x); 
 #else 
   return jn(n, x); 
 #endif 
 } 

template <class complex_t>
complex_t calculate_bessel_coeffs(int k, double x){};

template <>
cuComplex  calculate_bessel_coeffs<cuComplex>(int k, double x){
    return cuCmulf(imag_power(k), make_cuComplex(BesselJn(k, x), 0));
}

template <>
cuDoubleComplex calculate_bessel_coeffs<cuDoubleComplex>(int k, double x){
    return cuCmul(imag_power_fp64(k), make_cuDoubleComplex(BesselJn(k, x), 0));
}

double OneNorm(cuComplex* mat, unsigned int dim) { 
    double sum = 0; 
    double result = 0;
    for (int i = 0; i < dim; i++) { 
        sum = 0;
        
        for(int j = 0; j < dim; j++) {
            sum += cuCabsf(mat[dim*i+j]);
        }
        
        if (sum > result) {
            result = sum;
        }
    }
    return result;
} 

double OneNorm(cuDoubleComplex* mat, unsigned int dim) { 
    double sum = 0; 
    double result = 0;
    for (int i = 0; i < dim; i++) { 
        sum = 0;
        
        for(int j = 0; j < dim; j++) {
            sum += cuCabs(mat[dim*i+j]);
        }
        
        if (sum > result) {
            result = sum;
        }
    }
    return result;
} 


double OneNorm_fp64(cuDoubleComplex* arr, unsigned int dim){
    return OneNorm(arr,dim);
}