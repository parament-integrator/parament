import sys
import os
sys.path.insert(0,r'.\pyparament\src')
os.environ['PATH'] = os.getcwd() + '\\build\\' + ';' + os.environ['PATH']
import parament

import numpy as np

GPURunner = parament.Parament()