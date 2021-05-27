
[![Documentation Status](https://readthedocs.org/projects/parament/badge/?version=latest)](https://parament.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/parament-integrator/parament/actions/workflows/main.yml/badge.svg)](https://github.com/parament-integrator/parament/actions/workflows/main.yml)

# Parament
**Para**llelized **M**atrix **E**xponentiation for **N**umerical **T**ime evolution

Parament is a GPU-accelerated solver for time-dependent linear differential equations.
Notably, it solves the Schrödinger's equation with arbitrary time-dependent control terms.
Parament is open-source and is released under the Apache Licence.

## Documentation

The official documentation of Parament is [available here](https://parament.readthedocs.io/en/latest/).

## Installation

### Prerequisites (Windows)
1. Install Microsoft Visual Studio (Community Edition is sufficient)
2. Install the CUDA toolkit. Make sure the version of CUDA supports your GPU.
3. Add the MSVC compiler to your PATH. For Visual Studio 2019, it can be found at `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.26.28801\bin\Hostx64\x64`.

   You will have to substitute the exact version of the compiler you are using.
   
### Prerequisites (Linux)
CUDA toolkit must be installed.

### Building parament

*Note*: this step is for building the shared library (DLL). If you are only interested in using parament from Python,
skip ahead. The python installer will build the library automatically.

Run `build.bat` or `build.sh` in the `tools` folder.
The working directory must be set to the repository root.
The script will create a `/build` folder, containing the binaries.

Any extra command line arguments are forwarded
 directly to `nvcc`. Use this to e.g. to compile for an older
GPU architecture, that is not targeted by the default `nvcc` settings.   


### Building & installing pyparament

Pyparament is the official Python wrapper. It requires Python >=3.6.

At this time, pyparament is not yet available via PyPI. You can still install
the latest pyparament directly from Github by typing:
```
pip install git+https://github.com/parament-integrator/parament#subdirectory=src
```
This will automatically compile the C library, and install it alongside the Python wrapper.

If you need to pass special arguments to the `nvcc` compiler, you can do so by setting a `NVCC_ARGS` environment variable.
For instance, if you are targeting a Tesla K80 GPU (compute capability 3.7) from CUDA 11+, run (Windows and Linux, respectively):

```batch
set NVCC_ARGS=-arch=compute_37
pip install git+https://github.com/parament-integrator/parament#subdirectory=src
```

```bash
NVCC_ARGS="-arch=compute_37"
pip install git+https://github.com/parament-integrator/parament#subdirectory=src
```

