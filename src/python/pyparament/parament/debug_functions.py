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


import logging
import numpy as np

import parament.paramentlib
import parament.parament as parament


def expm(m):
    """Compute the exponential of the matrix `m`.

    Only works for matrices with small norms.
    """
    dt = 1.0
    gpu_runner = parament.Parament()
    gpu_runner.set_hamiltonian(1j * m, m, use_magnus=False, quadrature_mode='none')
    expm_gpu = gpu_runner.equiprop(dt, np.zeros(1))
    gpu_runner.destroy()
    return expm_gpu