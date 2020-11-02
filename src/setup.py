import os
from setuptools import setup, find_packages
import pathlib

USE_SHARED_PARAMENT = os.environ.get("USE_SHARED_PARAMENT")

CUDA_SRC_DIR = pathlib.Path(__file__).parent / '../src/cuda2'
CUDA_SRC_FILES = [f for f in os.listdir(CUDA_SRC_DIR) if os.path.isfile(os.path.join(CUDA_SRC_DIR, f))]

setup(
    name="parament",
    version="0.1",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    data_files=[('CUDA_SRC_DIR', CUDA_SRC_FILES)],
)
