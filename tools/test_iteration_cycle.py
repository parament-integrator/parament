import ctypes
import numpy as np
import timeit
import time
import knv as kNV
import matplotlib.pyplot as plt

lib = ctypes.cdll.LoadLibrary("build\parament.dll")

lib.Select_Iteration_cycles_fp32.argtypes = [ctypes.c_float,ctypes.c_float]
lib.Select_Iteration_cycles_fp32.restype = ctypes.c_int

out = lib.Select_Iteration_cycles_fp32(1,15)
print(out)

lib.OneNorm.argtypes = [np.ctypeslib.ndpointer(np.complex64,flags='F'),ctypes.c_int]
lib.OneNorm.restype = ctypes.c_float

mat = np.array([[1,1],[1,-1]])
mat = np.complex64(mat)
mat = np.asfortranarray(mat)
out = lib.OneNorm(mat,2)
print(out)

