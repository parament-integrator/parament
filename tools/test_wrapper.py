import sys
import os
sys.path.insert(0,r'.\pyparament\src')
os.environ['PATH'] = os.getcwd() + '\\build' + ';\r\n' + os.environ['PATH']
import parament

import numpy as np
import scipy.linalg


GPURunner = parament.Parament()
H0 = np.array([[1*100,0],[0,-1*100]])
H1 = np.array([[0,1],[1,0]])
dt = 0.02
carr = np.arange(200)/200
carr = np.array([carr[-1]])


expected = np.eye(2,dtype=np.complex128)
for i in range(len(carr)):
    H = (H0+carr[i]*H1)
    expected = scipy.linalg.expm(-1j*dt*H) @ expected
#expected = scipy.linalg.expm(-1j*dt*H0)

print("Expected: ")
print(expected)

GPURunner.setHamiltonian(H0,H1)

output_propagator = GPURunner.equiprop(carr,dt)

print("Calculated: ")
print(output_propagator)

diff = expected - output_propagator
diff = diff/(np.finfo(np.float32).eps)
print("Diff in units of eps")
print(diff)
#GPURunner.destroy()