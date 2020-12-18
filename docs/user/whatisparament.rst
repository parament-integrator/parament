.. _what-is-parament:

What is Parament?
=================

Parament (short for *Parallelized Matrix Exponentiation for Numeric Time integration*) is a GPU-accelerated solver for time-dependent linear differential equations.
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
time-dependence however, the problem generally has no analytical solution (instead, it involves `time-ordered exponentials <https://en.wikipedia.org/wiki/Ordered_exponential>`_).

Yet, this type of problem commonly appears in quantum mechanics. Indeed, the Schrödinger's equation for a spin in a magnetic field reads:

.. math::

    i\hbar \frac{\mathrm{d}}{\mathrm{d}t}|\Psi\rangle = (\hat{H_0} - \frac{\mu_B g}{\hbar}\hat{\vec{S}}\cdot \vec{B}(t))|\Psi\rangle

For simulating a time-varying magnetic field, the problem takes exactly the form described above!

Parament tackles this by slicing the problem into small time steps, during which the Hamiltonian is approximately constant.
Computing the propagator for every time slice is embarrassingly parallel, so GPUs are a natural choice.


