import parament
import numpy as np
import scipy.linalg

def test_fp32_expm_scipy_random(dim=2):
    np.random.seed(27)
    H0 = np.random.uniform(-1,1,(dim,dim))/dim + 1j*np.random.uniform(-1,1,(dim,dim))/2
    H0 = H0 + H0.conj().T
    H1 = np.zeros((dim,dim),dtype=complex)
    dt = 0.22285
    carr = np.zeros(1)
    GPURunner = parament.Parament()
    GPURunner.setHamiltonian(-H0,H1,use_magnus=False,quadrature_mode='just propagate')

    expm_GPU= GPURunner.equiprop(carr,dt)

    expm_SCIPY = scipy.linalg.expm(1j*dt*H0)
    error = np.sum(np.abs(expm_GPU-expm_SCIPY))
    error_threashold = np.finfo(np.float32).eps*dim*dim
    print(f"Error {error}")
    print(f"Error threashold {error_threashold}")

    error < error_threashold