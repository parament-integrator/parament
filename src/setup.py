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
    b" control_expansion.cu",
)
NVCC_BIN = b"nvcc"
NVCC_ARGS =[ 
    b"-lcublas",
    b"-DPARAMENT_BUILD_DLL",
    b"-DNDEBUG",  # disable assertions and debug messages
    b"--shared",
]



def run_nvcc(outputArgs):
    try:
        nvcc_cmd = [NVCC_BIN, *NVCC_ARGS, *outputArgs, *CUDA_SRC_FILES]
        print(b" ".join(nvcc_cmd).decode())
        subprocess.run(nvcc_cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to build CUDA code") from e


def run_nvcc_linux(outputArgs):
    try:
        NVCC_ARGS.append(b"--compiler-options -fPIC") # g++
        nvcc_cmd = [NVCC_BIN, *NVCC_ARGS, *outputArgs, *CUDA_SRC_FILES]
        print(b" ".join(nvcc_cmd).decode())
        linux_command = b" ".join(nvcc_cmd).decode()
        subprocess.run(linux_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to build CUDA code") from e


def build_parament_cuda(buildDir: pathlib.Path):
    #
    cwd = os.getcwd()
    buildDir = buildDir.absolute()  # make path absolute, we are about to change the cwd
    try:
        os.chdir(CUDA_SRC_DIR)

        if platform.system() == "Windows":
            outputArgs = [b"-o", str(buildDir / "parament.dll").encode()]
            run_nvcc(outputArgs)
            # remove EXP and LIB files, keep only DLL
            os.remove(str(buildDir / "parament.exp"))
            os.remove(str(buildDir / "parament.lib"))
        elif platform.system() == "Linux":
            os.path.dirname(CUDA_SRC_DIR)
            #raise Exception(os.listdir())
            outputArgs = [b"-o", str(buildDir / "libparament.so").encode()]
            run_nvcc_linux(outputArgs)
            print(buildDir)
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


setup(
    name="parament",
    version="0.1",
    package_dir={'': 'python/pyparament'},
    packages=find_packages(where='python/pyparament'),
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
        ]
    }
)
