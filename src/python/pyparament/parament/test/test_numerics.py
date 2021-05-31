# Copyright 2021 Konstantin Herb, Pol Welter. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ##########################################################################


import parament
import numpy as np
import scipy.linalg
import pytest

NUMERICS_LOGFILE = 'numerics.log'


def random_hamiltonians(dim):
    np.random.seed(27)
    H0 = np.random.uniform(-1, 1, (dim, dim)) / dim + 1j * np.random.uniform(-1, 1, (dim, dim)) / 2
    H0 = H0 + H0.conj().T
    H1 = np.zeros((dim, dim), dtype=complex)
    return H0, H1


@pytest.mark.parametrize("dim", [2, 16])
def test_fp32_expm_scipy_random(dim):
    H0, H1 = random_hamiltonians(dim)
    dt = 0.01
    carr = np.zeros(1)

    with parament.Parament() as context:
        context.set_hamiltonian(H0, H1, use_magnus=False, quadrature_mode='none')
        expm_gpu = context.equiprop(dt, carr)

    expm_scipy = scipy.linalg.expm(-1j*dt*H0)

    error = np.sum(np.abs(expm_gpu-expm_scipy))
    eps = np.finfo(np.float32).eps
    error_threshold = eps*dim*dim
    msg = f"""
    --------------------------------
    FP32: Difference from scipy, for a {dim}x{dim} matrix: {error}, machine epsilon {eps}
    """
    f = open(NUMERICS_LOGFILE, 'a')
    f.write(msg)
    f.close()
    print(f"FP32 Error {error}")
    print(f"FP32 Error threshold {error_threshold}")

    assert error < error_threshold


@pytest.mark.parametrize("dim", [2, 4])
def test_fp32_expm_debug(dim):
    H0, _ = random_hamiltonians(dim)
    expm_gpu = parament.debug_functions.expm(H0)
    expm_scipy = scipy.linalg.expm(H0)
    error = np.sum(np.abs(expm_gpu-expm_scipy))
    eps = np.finfo(np.float32).eps
    error_threshold = eps*dim*dim

    assert error < error_threshold


@pytest.mark.parametrize("dim", [2, 16])
def test_fp64_expm_scipy_random(dim):
    H0, H1 = random_hamiltonians(dim)
    dt = 0.01
    carr = np.zeros(1)

    with parament.parament.Parament(precision='fp64')as context:
        context.set_hamiltonian(H0, H1, use_magnus=False, quadrature_mode='none')

        expm_gpu = context.equiprop(dt, carr)

    expm_SCIPY = scipy.linalg.expm(-1j*dt*H0)
    error = np.sum(np.abs(expm_gpu-expm_SCIPY))
    eps = np.finfo(np.float64).eps
    error_threshold = eps*dim*dim
    msg = f"""
    --------------------------------
    FP64: Difference from scipy, for a {dim}x{dim} matrix: {error}, machine epsilon {eps}
    """
    f = open(NUMERICS_LOGFILE, 'a')
    f.write(msg)
    f.close()
    print(f"FP64 Error {error}")
    print(f"FP64 Error threshold {error_threshold}")

    assert error < error_threshold


def test_fp32_multi_fields():
    dtype_base = np.complex128

    ox = np.array([[0, 1], [1, 0]], dtype=dtype_base)
    oy = np.array([[0, -1j], [1j, 0]], dtype=dtype_base)
    oz = np.array([[1, 0], [0, -1]], dtype=dtype_base)
    w0 = 1.0
    w1 = 0.1
    H0 = w0 / 2 * oz
    H1 = w1 / 2 * ox
    H2 = w1 / 2 * oy

    carr1 = np.array([0.1, 0.2])
    carr2 = np.array([0.3, 0.4])
    dt = 0.1
    with parament.Parament(precision='fp32') as context:
        context.set_hamiltonian(H0, H1, H2, use_magnus=False, quadrature_mode='none')
        propagator = context.equiprop(dt, carr1, carr2)

    U = np.eye(2, dtype=np.complex128)
    for i in range(len(carr1)):
        X = H0 + carr1[i] * H1 + carr2[i] * H2
        G = -1j * X * dt
        print(X)

        print("------------")
        print(scipy.linalg.expm(G))
        U = scipy.linalg.expm(G) @ U
        print("##############")
    assert np.linalg.norm(U - propagator) < 1e-6
