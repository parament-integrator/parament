import logging
import numpy as np

import parament.paramentlib
import parament.parament as parament

def expm_debug(m):
    dt = -1
    carr = np.zeros(1)
    GPURunner = parament.Parament()
    GPURunner.setHamiltonian(-m,m,use_magnus=False,quadrature_mode='just propagate')
    expm_GPU= GPURunner.equiprop(carr,dt)
    GPURunner.destroy()
    return expm_GPU