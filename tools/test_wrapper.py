import sys
import os
sys.path.insert(0,r'.\pyparament\src')
os.environ['PATH'] = os.getcwd() + '\\build' + ';\r\n' + os.environ['PATH']
import parament

import numpy as np
import scipy.linalg


GPURunner = parament.Parament()
H0 = np.array([[1,0],[0,-1]])
H1 = np.array([[0,1],[1,0]])
dt = 1.0

print("Expected: ")
print(scipy.linalg.expm(-1j*dt*H0))

GPURunner.setHamiltonian(H0,H1)

output_propagator = GPURunner.equiprop(np.zeros(1),dt)

print(output_propagator)
#GPURunner.destroy()
