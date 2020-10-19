import sys
import os
sys.path.insert(0,r'.\pyparament\src')
os.environ['PATH'] = os.getcwd() + '\\build\\' + ';' + os.environ['PATH']
import parament

import numpy as np

GPURunner = parament.Parament()
H0 = np.array([[1,0],[0,-1]])
H1 = np.array([[0,1],[1,0]])

GPURunner.setHamiltonian(H0,H1)