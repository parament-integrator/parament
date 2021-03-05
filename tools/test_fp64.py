import sys
import os
sys.path.append(os.path.abspath(r'./src/python/pyparament/'))
os.environ['PARAMENT_LIB_DIR'] = os.path.abspath(r'./build')

import sys
import pprint

# pretty print module search paths
pprint.pprint(sys.path)


import parament

import numpy as np
import scipy.linalg

print("HALLO. JETZT GEHTS LOS!")
GPURunner = parament.Parament(precision='fp64')
H0 = np.array([[1*100,0],[0,-1*100]])
H1 = np.array([[0,1],[1,0]])
dt = 0.02
carr = np.arange(40000)/40000
#carr = np.array([carr[-1]])


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
diff = diff/(np.finfo(np.float64).eps)
print("Diff in units of eps")
print(diff)
#GPURunner.destroy()