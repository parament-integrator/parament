import ctypes
import numpy as np
import timeit
import time
import knv as kNV
import matplotlib.pyplot as plt

kokoDLL = ctypes.WinDLL("hello.dll")

lib = ctypes.cdll.LoadLibrary('hello.dll')
class GPURunner(object):
    def __init__(self):
        self.dim = 0
        lib.GPURunner_new.argtypes = []
        lib.GPURunner_new.restype = ctypes.c_void_p
        #lib.GPURunner_bar.argtypes = [ctypes.c_void_p]
        #lib.GPURunner_bar.restype = ctypes.c_void_p
        #lib.GPURunner_test.argtypes = [ctypes.c_void_p,np.ctypeslib.ndpointer(np.float32), np.ctypeslib.ndpointer(np.float32), ctypes.c_uint] 
        #lib.GPURunner_test.restype = ctypes.c_void_p
        #lib.GPURunner_propagate.argtypes = [ctypes.c_void_p, np.ctypeslib.ndpointer(np.complex64), np.ctypeslib.ndpointer(np.complex64), np.ctypeslib.ndpointer(np.complex64), np.ctypeslib.ndpointer(np.complex64), np.ctypeslib.ndpointer(np.float32), np.ctypeslib.ndpointer(np.float32), ctypes.c_uint, ctypes.c_uint] 
        #lib.GPURunner_propagate.restype = ctypes.c_void_p
        
        lib.GPURunner_readback.argtypes = [ctypes.c_void_p]
        lib.GPURunner_readback.restype = ctypes.c_void_p
        lib.GPURunner_sethamiltonian.argtypes = [ctypes.c_void_p, np.ctypeslib.ndpointer(np.complex64,flags='F'), np.ctypeslib.ndpointer(np.complex64,flags='F'), ctypes.c_uint] 
        lib.GPURunner_sethamiltonian.restype = ctypes.c_void_p
        
        lib.GPURunner_equiprop.argtypes = [ctypes.c_void_p, np.ctypeslib.ndpointer(np.complex64,flags='F'), ctypes.c_float, ctypes.c_uint, np.ctypeslib.ndpointer(np.complex64,flags='F')]
        lib.GPURunner_equiprop.restype = ctypes.c_void_p
        
        self.obj = lib.GPURunner_new()
        
    def __del__(self):
        lib.GPURunner_del.argtypes = [ctypes.c_void_p]
        lib.GPURunner_del.restype = ctypes.c_void_p
        lib.GPURunner_del(self.obj)
        
    
    def propagate(self,H0,H1,carr,dtarr):
        dim = np.shape(H0)
        dim = dim[0]
        pts = len(carr)
        #reH0 =np.float32(np.real(H0))
        #imH0 =np.float32(np.imag(H0))
        #reH1 =np.float32(np.real(H1))
        #imH1 =np.float32(np.imag(H1))
        #lib.GPURunner_propagate(self.obj,H0,H0,H1,H1,carr,dtarr,dim,pts)
    
    def set_hamiltonian(self,H0,H1):
        dim = np.shape(H0)
        dim = dim[0]
        self.dim = dim
        lib.GPURunner_sethamiltonian(self.obj,np.asfortranarray(H0),np.asfortranarray(H1),dim)
        
    def readback(self):
        lib.GPURunner_readback(self.obj)
    
    def equiprop(self, c_arr, dt):
        pts = len(c_arr)
        #print(np.asfortranarray(c_arr))
        output = np.empty(self.dim**2,dtype=np.complex64,order='F')
        lib.GPURunner_equiprop(self.obj,np.asfortranarray(c_arr),dt,pts,output)
        return output
        

neu = GPURunner()
id3 = np.identity(3,dtype=np.complex64)
id2 = np.identity(2,dtype=np.complex64)
H0 = np.complex64(np.array([[0,1],[1,0]]))
H0 = np.kron(id3,H0)
H0 = np.kron(id2,H0)
#H0 = np.kron(id2,H0)
#H0 = np.kron(id2,H0)


H1 = H0
#carr = np.complex64(np.array([0,0,1,0,0,0,0,0,0]))
carr = np.complex64(np.zeros(80000))

dt = np.pi/4
neu.set_hamiltonian(H0,H1)
    
print(np.shape(H0))
for i in range(100):
    out_gpu = neu.equiprop(carr,dt)

del neu

