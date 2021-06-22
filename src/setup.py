# Copyright 2021 Konstantin Herb, Pol Welter. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ##########################################################################


import os
from setuptools import setup, find_packages
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
import distutils.command.build
import pathlib
import platform
import subprocess

USE_SHARED_PARAMENT = os.environ.get("USE_SHARED_PARAMENT")
SRC_DIR = pathlib.Path(__file__).parent
CUDA_SRC_DIR = SRC_DIR / 'cuda'

CUDA_SRC_FILES = (
    b"deviceInfo.c",
    b"diagonal_add.cu",
    b"mathhelper.cpp",
    b"parament.cpp",
    b"printFuncs.cpp",
    b"debugfuncs.cpp",
    b"control_expansion.cu",
)
NVCC_BIN = b"nvcc"
NVCC_ARGS = [
    b"-lcublas",
    b"-DPARAMENT_BUILD_DLL",
    b"--shared",
]
if os.environ.get("NVCC_ARGS"):
    NVCC_USER_ARGS = [os.environ.get("NVCC_ARGS").encode()]
else:
    NVCC_USER_ARGS = [
        b"-DNDEBUG",  # disable assertions and debug messages
    ]


def run_nvcc_win(build_dir: pathlib.Path):
    try:
        nvcc_cmd = [
            NVCC_BIN,
            *NVCC_ARGS,
            *NVCC_USER_ARGS,
            b"-o", str(build_dir / "parament.dll").encode(),
            *CUDA_SRC_FILES]
        subprocess.run(nvcc_cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to build CUDA code") from e


def run_nvcc_linux(build_dir: pathlib.Path):
    try:
        nvcc_cmd = [
            NVCC_BIN,
            *NVCC_ARGS,
            *NVCC_USER_ARGS,
            b"--compiler-options -fPIC",
            b"-o", str(build_dir / "libparament.so").encode(),
            *CUDA_SRC_FILES
        ]
        linux_command = b" ".join(nvcc_cmd).decode()
        subprocess.run(linux_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to build CUDA code") from e


def build_parament_cuda(build_dir: pathlib.Path):
    cwd = os.getcwd()
    build_dir = build_dir.absolute()  # make path absolute, we are about to change the cwd
    try:
        os.chdir(CUDA_SRC_DIR)

        if platform.system() == "Windows":
            run_nvcc_win(build_dir)
            # remove EXP and LIB files, keep only DLL
            os.remove(str(build_dir / "parament.exp"))
            os.remove(str(build_dir / "parament.lib"))
        elif platform.system() == "Linux":
            run_nvcc_linux(build_dir)
        else:
            raise RuntimeError("Don't know how to build on " + platform.system())
    finally:
        os.chdir(cwd)  # restore original working directory


# Override build command
class BuildCommand(distutils.command.build.build):
    def run(self):
        # Run the original build command
        distutils.command.build.build.run(self)

        if USE_SHARED_PARAMENT is None:
            buildDir = pathlib.Path(self.build_lib) / "parament"
            if not os.path.exists(buildDir):
                os.makedirs(buildDir)
            build_parament_cuda(buildDir)


# Override bdist_wheel command, to force the wheel to be platform specific.
# https://stackoverflow.com/questions/45150304/how-to-force-a-python-wheel-to-be-platform-specific-when-building-it
class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        if USE_SHARED_PARAMENT is None:
            # Mark us as not a pure python package
            self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = _bdist_wheel.get_tag(self)
        python, abi = 'py3', 'none'
        return python, abi, plat

    
with open(SRC_DIR / "python/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

    
setup(
    name="parament",
    version="0.1.0",
    author="Konstantin Herb, Pol Welter",
    author_email="science@rashbw.de",
    description="Parament Integrator",
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_dir={'': 'python/pyparament'},
    packages=find_packages(where='python/pyparament'),
    url="https://github.com/parament-integrator/parament",
    project_urls={
        "Documentation": "https://parament.readthedocs.io/en/latest/",
        "Bug Tracker": "https://github.com/parament-integrator/parament/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    cmdclass={
        "build": BuildCommand,
        "bdist_wheel": bdist_wheel,
    },
    install_requires=[
        'numpy'
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
            'scipy',
        ]
    },
    python_requires=">=3.6",
)
