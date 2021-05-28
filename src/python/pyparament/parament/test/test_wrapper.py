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


def test_smoke():
    gpu_runner = parament.Parament()
    H0 = np.array([[1, 0], [0, -1]])
    H1 = np.array([[0, 1], [1, 0]])

    gpu_runner.set_hamiltonian(H0, H1)
    gpu_runner.destroy()


def test_simple():
    gpu_runner = parament.Parament()
    H0 = np.array([[1, 0], [0, -1]])
    H1 = np.array([[0, 1], [1, 0]])
    dt = 1.0

    expected_matrix_exponential = scipy.linalg.expm(-1j * dt * H0)

    print("Expected: ")
    print(expected_matrix_exponential)

    gpu_runner.set_hamiltonian(H0, H1)

    output_propagator = gpu_runner.equiprop(dt, np.zeros(1))
    gpu_runner.destroy()

    print("Calculated: ")
    print(output_propagator)

    diff = expected_matrix_exponential - output_propagator
    diff = diff / np.finfo(np.float32).eps
    print("Diff in units of eps")
    print(diff)

