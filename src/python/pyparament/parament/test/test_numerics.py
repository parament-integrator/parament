import parament
import numpy as np
import scipy.linalg

NUMERICS_LOGFILE = 'numerics.log'


def test_fp32_expm_scipy_random(dim=2):
    np.random.seed(27)
    H0 = np.random.uniform(-1, 1, (dim, dim))/dim + 1j*np.random.uniform(-1, 1, (dim,dim))/2
    H0 = H0 + H0.conj().T
    H1 = np.zeros((dim, dim), dtype=complex)
    dt = 0.22285
    carr = np.zeros(1)
    gpu_runner = parament.Parament()
    gpu_runner.setHamiltonian(-H0, H1, use_magnus=False, quadrature_mode='none')

    expm_GPU = gpu_runner.equiprop(carr, dt)

    expm_SCIPY = scipy.linalg.expm(1j*dt*H0)
    error = np.sum(np.abs(expm_GPU-expm_SCIPY))
    eps = np.finfo(np.float32).eps
    error_threshold = eps*dim*dim
    msg = f"""
    --------------------------------
    FP32: Difference from scipy, for a {dim}x{dim} matrix: {error}, machine epsilon {eps}
    """
    f = open(NUMERICS_LOGFILE,'a')
    f.write(msg)
    f.close()
    print(f"FP32 Error {error}")
    print(f"FP32 Error threashold {error_threshold}")

    assert error < error_threshold
