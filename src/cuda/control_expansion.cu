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


#include <cublas_v2.h>
#include <stdlib.h>
#include <stdio.h>
#define NO_CUDA_STUBS
#include "control_expansion.h"

// 3D thread block indexing
__global__ void generate_magnus(cuComplex *coeffs_in, cuComplex *coeffs_out, int amps, int n, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Timepoint index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Amp index
    int k = blockIdx.z * blockDim.z + threadIdx.z;  //Second amp index
    int len_new = (n-3)/2+1;
    cuComplex four = make_cuComplex(4,0);
    cuComplex six = make_cuComplex(6,0);
    
    // commiefactor = i*dt/12
    cuComplex commiefactor = cuCdivf(make_cuComplex(0,1),make_cuComplex(12,0));
    commiefactor = cuCmulf(make_cuComplex(dt,0),commiefactor);
    
    if (i < (n-3)/2+1 && j < amps && k == 0){
        coeffs_out[i+j*len_new] = cuCaddf(coeffs_in[2*i+j*n], cuCmulf(four,coeffs_in[2*i+j*n+1]));
        coeffs_out[i+j*len_new] = cuCaddf(coeffs_out[i+j*len_new], coeffs_in[2*i+j*n+2]);
        coeffs_out[i+j*len_new] = cuCdivf(coeffs_out[i+j*len_new], six);

        // Coefficients for commutator [H0,control H]
        int idx_comm = i+(j+amps)*len_new;
        coeffs_out[idx_comm] = cuCsubf(coeffs_in[2*i+j*n+2],coeffs_in[2*i+j*n]);
        coeffs_out[idx_comm] = cuCmulf(coeffs_out[idx_comm],commiefactor);
        
    

    }


    /*
    // Coefficient calculations involving only one control Hamiltonian
    if (i < len_new && j < amps && k==0){

        // Coefficient reduction for Simpson rule
        int idx_new = i+j*len_new;
        int idx_old = 2*i+n*j;
        //coeff_out[idx_new] = 1/6*coeff_in[idx_old] + 4/6*coeff_in[idx_old+1] + 1/6*coeff_in[idx_old+2];
        coeffs_out[idx_new] = cuCaddf(coeffs_in[idx_old], cuCmulf(four,coeffs_in[idx_old+1]));
        coeffs_out[idx_new] = cuCaddf(coeffs_in[idx_new], coeffs_in[idx_old+2]);
        coeffs_out[idx_new] = cuCdivf(coeffs_in[idx_new], six);

    }
    */

    
    // Coefficient calculations for pairwise commutators of control Hamiltonians
    if (i < len_new && j < k && k < amps){
        int idx_old_amp1 = 2*i+n*j;
        int idx_old_amp2 = 2*i+n*k;
        int idx_new_pair = i+j*len_new+(k-1)*len_new + len_new*amps*2;
        //coeff_out[idx_new_pair] = coeff_in[idx_old_amp1]*coeff_in[idx_old_amp2+2]-coeff_in[idx_old_amp1+2]*coeff_in[idx_old_amp2];
        coeffs_out[idx_new_pair] = cuCmulf(coeffs_in[idx_old_amp1],coeffs_in[idx_old_amp2+2]);
        coeffs_out[idx_new_pair] = cuCsubf(coeffs_out[idx_new_pair],cuCmulf(coeffs_in[idx_old_amp1+2],coeffs_in[idx_old_amp2]));
        coeffs_out[idx_new_pair] = cuCmulf(coeffs_out[idx_new_pair],commiefactor);
        

    }
    

}


__global__ void generate_magnus_fp64(cuDoubleComplex *coeffs_in, cuDoubleComplex *coeffs_out, int amps, int n, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Timepoint index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Amp index
    int k = blockIdx.z * blockDim.z + threadIdx.z;  //Second amp index
    int len_new = (n-3)/2+1;
    cuDoubleComplex four = make_cuDoubleComplex(4,0);
    cuDoubleComplex six = make_cuDoubleComplex(6,0);
    
    // commiefactor = i*dt/12
    cuDoubleComplex commiefactor = cuCdiv(make_cuDoubleComplex(0,1),make_cuDoubleComplex(12,0));
    commiefactor = cuCmul(make_cuDoubleComplex(dt,0),commiefactor);
    
    if (i < (n-3)/2+1 && j < amps && k == 0){
        coeffs_out[i+j*len_new] = cuCadd(coeffs_in[2*i+j*n], cuCmul(four,coeffs_in[2*i+j*n+1]));
        coeffs_out[i+j*len_new] = cuCadd(coeffs_out[i+j*len_new], coeffs_in[2*i+j*n+2]);
        coeffs_out[i+j*len_new] = cuCdiv(coeffs_out[i+j*len_new], six);

        // Coefficients for commutator [H0,control H]
        int idx_comm = i+(j+amps)*len_new;
        coeffs_out[idx_comm] = cuCsub(coeffs_in[2*i+j*n+2],coeffs_in[2*i+j*n]);
        coeffs_out[idx_comm] = cuCmul(coeffs_out[idx_comm],commiefactor);
        
    

    }

    
    // Coefficient calculations for pairwise commutators of control Hamiltonians
    if (i < len_new && j < k && k < amps){
        int idx_old_amp1 = 2*i+n*j;
        int idx_old_amp2 = 2*i+n*k;
        int idx_new_pair = i+j*len_new+(k-1)*len_new + len_new*amps*2;
        //coeff_out[idx_new_pair] = coeff_in[idx_old_amp1]*coeff_in[idx_old_amp2+2]-coeff_in[idx_old_amp1+2]*coeff_in[idx_old_amp2];
        coeffs_out[idx_new_pair] = cuCmul(coeffs_in[idx_old_amp1],coeffs_in[idx_old_amp2+2]);
        coeffs_out[idx_new_pair] = cuCsub(coeffs_out[idx_new_pair],cuCmul(coeffs_in[idx_old_amp1+2],coeffs_in[idx_old_amp2]));
        coeffs_out[idx_new_pair] = cuCmul(coeffs_out[idx_new_pair],commiefactor);
        

    }
    

}



// 2D thread block indexing
__global__ void generate_midpoint(cuComplex *coeffs_in, cuComplex *coeffs_out, int amps, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Timepoint index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Control field index 
    cuComplex half = make_cuComplex(0.5,0);

    if (i < n - 1 && j < amps){
        coeffs_out[i+j*n] = cuCaddf(coeffs_in[i+j*n],coeffs_in[i+j*n+1]);
        coeffs_out[i+j*n] = cuCmulf(half,coeffs_out[i+j*n]);
    }
}

// 2D thread block indexing
__global__ void generate_midpoint_fp64(cuDoubleComplex *coeffs_in, cuDoubleComplex *coeffs_out, int amps, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Timepoint index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Control field index 
    cuDoubleComplex half = make_cuDoubleComplex(0.5,0);

    if (i < n - 1 && j < amps){
        coeffs_out[i+j*n] = cuCadd(coeffs_in[i+j*n],coeffs_in[i+j*n+1]);
        coeffs_out[i+j*n] = cuCmul(half,coeffs_out[i+j*n]);
    }
}

// 2D thread block indexing
__global__ void generate_simpson(cuComplex *coeffs_in, cuComplex *coeffs_out, int amps, int n)
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // Timepoint index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Control field index 
    cuComplex four = make_cuComplex(4,0);
    cuComplex six = make_cuComplex(6,0);
    int len_new = (n-3)/2+1;

    if (i < (n-3)/2+1 && j < amps){
        coeffs_out[i+j*len_new] = cuCaddf(coeffs_in[2*i+j*n], cuCmulf(four,coeffs_in[2*i+j*n+1]));
        coeffs_out[i+j*len_new] = cuCaddf(coeffs_out[i+j*len_new], coeffs_in[2*i+j*n+2]);
        coeffs_out[i+j*len_new] = cuCdivf(coeffs_out[i+j*len_new], six);
    

    }
}


__global__ void generate_simpson_fp64(cuDoubleComplex *coeffs_in, cuDoubleComplex *coeffs_out, int amps, int n)
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // Timepoint index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Control field index 
    cuDoubleComplex four = make_cuDoubleComplex(4,0);
    cuDoubleComplex six = make_cuDoubleComplex(6,0);
    int len_new = (n-3)/2+1;

    if (i < (n-3)/2+1 && j < amps){
        coeffs_out[i+j*len_new] = cuCadd(coeffs_in[2*i+j*n], cuCmul(four,coeffs_in[2*i+j*n+1]));
        coeffs_out[i+j*len_new] = cuCadd(coeffs_out[i+j*len_new], coeffs_in[2*i+j*n+2]);
        coeffs_out[i+j*len_new] = cuCdiv(coeffs_out[i+j*len_new], six);
    

    }
}


// ////////////////////////////////////////
// HERE FUNCTIONS EXPOSED TO C/C++ FOLLOW
// ///////////////////////////////////////


void control_magnus(cuComplex* coeff_in, cuComplex *coeff_out, unsigned int amps, unsigned int n, float dt, unsigned int numSMs)
{
    dim3 threadsPerBlock(256/(amps*amps), amps,amps);
    dim3 numBlocks(n / threadsPerBlock.x+1, 1);
    generate_magnus<<<numBlocks, threadsPerBlock>>>(coeff_in, coeff_out, amps, n, dt);
}

void control_magnus(cuDoubleComplex* coeff_in, cuDoubleComplex *coeff_out, unsigned int amps, unsigned int n, float dt, unsigned int numSMs)
{
    dim3 threadsPerBlock(256/(amps*amps), amps,amps);
    dim3 numBlocks(n / threadsPerBlock.x+1, 1);
    generate_magnus_fp64<<<numBlocks, threadsPerBlock>>>(coeff_in, coeff_out, amps, n, dt);
}


void control_midpoint(cuComplex* coeff_in, cuComplex *coeff_out, unsigned int amps, unsigned int n, unsigned int numSMs)
{
    dim3 threadsPerBlock(256/amps, amps);
    dim3 numBlocks(n / threadsPerBlock.x+1, 1);
    generate_midpoint<<<numBlocks, threadsPerBlock>>>(coeff_in, coeff_out, amps, n);
}

void control_midpoint(cuDoubleComplex* coeff_in, cuDoubleComplex *coeff_out, unsigned int amps, unsigned int n, unsigned int numSMs)
{
    dim3 threadsPerBlock(256/amps, amps);
    dim3 numBlocks(n / threadsPerBlock.x+1, 1);
    generate_midpoint_fp64<<<numBlocks, threadsPerBlock>>>(coeff_in, coeff_out, amps, n);
}


void control_simpson(cuComplex* coeff_in, cuComplex *coeff_out, unsigned int amps, unsigned int n, unsigned int numSMs)
{
    dim3 threadsPerBlock(256/amps, amps);
    dim3 numBlocks((n-3)/2 / threadsPerBlock.x+1, 1);
    generate_simpson<<<numBlocks, threadsPerBlock>>>(coeff_in, coeff_out, amps, n);
}

void control_simpson(cuDoubleComplex* coeff_in, cuDoubleComplex *coeff_out, unsigned int amps, unsigned int n, unsigned int numSMs)
{
    dim3 threadsPerBlock(256/amps, amps);
    dim3 numBlocks((n-3)/2 / threadsPerBlock.x+1, 1);
    generate_simpson_fp64<<<numBlocks, threadsPerBlock>>>(coeff_in, coeff_out, amps, n);
}