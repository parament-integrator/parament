import parament
import numpy as np
import scipy.linalg


def test_smoke():
    GPURunner = parament.Parament()
    H0 = np.array([[1,0], [0,-1]])
    H1 = np.array([[0,1], [1,0]])

    GPURunner.setHamiltonian(H0, H1)
    GPURunner.destroy()


def test_simple():
    GPURunner = parament.Parament()
    H0 = np.array([[1, 0], [0, -1]])
    H1 = np.array([[0, 1], [1, 0]])
    dt = 1.0

    expected_matrix_exponential = scipy.linalg.expm(-1j * dt * H0)

    print("Expected: ")
    print(expected_matrix_exponential)

    GPURunner.setHamiltonian(H0, H1)

    output_propagator = GPURunner.equiprop(np.zeros(1), dt)
    GPURunner.destroy()

    print("Calculated: ")
    print(output_propagator)

    diff = expected_matrix_exponential - output_propagator
    diff = diff / (np.finfo(np.float32).eps)
    print("Diff in units of eps")
    print(diff)
