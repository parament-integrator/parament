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


def expm_debug(m):
    dt = -1
    carr = np.zeros(1)
    GPURunner = parament.Parament()
    GPURunner.setHamiltonian(-m,m,use_magnus=False,quadrature_mode='none')
    expm_GPU= GPURunner.equiprop(carr,dt)
    GPURunner.destroy()
    return expm_GPU