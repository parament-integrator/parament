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

// Returns the kth power of the imaginary unit in the cuComplex type
cuComplex imag_power(int k) {
    if      (k%4 == 0){
        return make_cuComplex(1, 0);
    }
    else if (k%4 ==1 ){
        return make_cuComplex(0, -1);
    }
    else if (k%4 == 2){
        return make_cuComplex(-1, 0);
    }
    else {
        return make_cuComplex(0, 1);
    }
}

// Returns the Bessel function array as cuComplex type
void J_arr(cuComplex* arr, int mmax, double c) {
    int i = 0;
    for (i = 0; i < mmax + 1; i++) {
        arr[i] = cuCmulf(imag_power(i),make_cuComplex(_jn(i,c), 0));
    }
    return;
}
