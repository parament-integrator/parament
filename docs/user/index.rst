.. _user-guide:

Parament User guide
###################

This is the user's guide of Parament. Here you can find instructions and tutorials on how to use Parament.

If you are already familiar with Parament, and are looking for details of the API, head over to the :ref:`Reference<api-reference>`.

.. _what-is-parament:

What is Parament?
=================

Parament (short for *Parallelized Matrix Exponentiation for Numeric Time integration*) is a GPU-accelerated solver for
time-dependent linear differential equations.
Notably, it solves the Schrödinger's equation with arbitrary time-dependent control terms.
Parament is open-source and is released under the Apache Licence.

What does it do?
----------------

Parament is a solver for differential equations of the form

.. math::

    \frac{\mathrm{d}}{\mathrm{d}t}x = \left(A  + \sum_{k=1}^N B_k u_k(t)\right)x,

where :math:`x\in\mathbb{C}^n` is the state vector, :math:`A, B_1, ..., B_N\in\mathbb{C}^{n\times n}`, and
:math:`u_k(t)\in\mathbb{C}`.

If :math:`u_k` was not time-dependent, the solution would be trivial: a simple matrix exponential. With the added
time-dependence however, the problem generally has no analytical solution (instead, it involves `time-ordered
exponentials <https://en.wikipedia.org/wiki/Ordered_exponential>`_).

Yet, this type of problem commonly appears in quantum mechanics. Indeed, the Schrödinger's equation for a spin in a
magnetic field reads:

.. math::

    i\hbar \frac{\mathrm{d}}{\mathrm{d}t}|\Psi\rangle = (\hat{H_0} - \frac{\mu_B g}{\hbar}\hat{\vec{S}}\cdot \vec{B}(t))|\Psi\rangle

For simulating a time-varying magnetic field, the problem takes exactly the form described above!

Parament tackles this by slicing the problem into small time steps, during which the Hamiltonian is approximately constant.
Computing the propagator for every time slice is embarrassingly parallel, so GPUs are a natural choice.


Research paper
--------------

If you want to know the nitty-gritty details, have a look at the research paper which got published in *Computer Physics Communications*:

| Konstantin Herb, Pol Welter:
| *Parallel time integration using Batched BLAS (Basic Linear Algebra Subprograms) routines*
| Computer Physics Communications **270**, 2022, 108181, ISSN 0010-4655
| https://doi.org/10.1016/j.cpc.2021.108181

Components
==========

Parament uses the CUDA framework. Hence, it requires a CUDA compatible GPU to be present in the system.

At its core, Parament is a C library, that your application code can dynamically link to. It can be used from within
any programming language that supports calling into shared libraries.

We provide a wrapper for the Python programming language. In the future, more official wrappers for other languages
(e.g. Matlab) might become available.

Throughout this documentation, we refer to both the core library and the Python wrapper as *Parament*. Occasionally, we
will use the more specific terms *paramentlib* and *pyparament*, if necessary.


.. _installation:

Installation
============

Prerequisites (Windows)
-----------------------

1. Install Microsoft Visual Studio (Community Edition is sufficient)
2. Install the CUDA toolkit. Make sure the version of CUDA supports your GPU.
3. Add the MSVC compiler to your PATH. For Visual Studio 2019, it can be found at
   ``C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.26.28801\bin\Hostx64\x64``.

   You will have to substitute the exact version of the compiler you are using.

Prerequisites (Linux)
---------------------
CUDA toolkit must be installed.

Building parament
-----------------

*Note*: this step is for building the shared library (DLL). If you are only interested in using parament from Python,
skip ahead. The python installer will build the library automatically.

Run ``build.bat`` or ``build.sh`` in the `tools` folder.
The working directory must be set to the repository root.
The script will create a ``/build`` folder, containing the binaries.

Any extra command line arguments are forwarded
directly to ``nvcc``. Use this to e.g. to compile for an older
GPU architecture, that is not targeted by the default ``nvcc`` settings.


Building & installing pyparament
--------------------------------

Pyparament is the official Python wrapper. It requires Python >=3.6.

At this time, pyparament is not yet available via PyPI. You can still install
the latest pyparament directly from Github by typing:

.. code-block::

    pip install git+https://github.com/parament-integrator/parament#subdirectory=src

This will automatically compile and install a bundled copy of `paramentlib`.

If you need to pass special arguments to the ``nvcc`` compiler, you can do so by setting a ``NVCC_ARGS`` environment
variable. For instance, if you are targeting a Tesla K80 GPU (compute capability 3.7) from CUDA 11+, run (Windows and
Linux, respectively):

.. code-block:: batch

    set NVCC_ARGS=-arch=compute_37
    pip install git+https://github.com/parament-integrator/parament#subdirectory=src


.. code-block:: bash

    NVCC_ARGS="-arch=compute_37"
    pip install git+https://github.com/parament-integrator/parament#subdirectory=src



