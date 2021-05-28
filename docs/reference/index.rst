.. _api-reference:

API Reference
#############

This part of the documentation lists all the functions provided by Parament.

C API
-----
Functions
~~~~~~~~~

.. autocmodule:: cuda/parament.h
   :members: Parament_create, Parament_destroy, Parament_setHamiltonian, Parament_equiprop

.. autocmodule:: cuda/parament.h
   :members: Parament_selectIterationCycles_fp32, Parament_setIterationCyclesManually, Parament_automaticIterationCycles

.. autocmodule:: cuda/parament.h
    :members: Parament_peekAtLastError, Parament_errorMessage

Double-precision functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autocmodule:: cuda/parament.h
   :members: Parament_create_fp64, Parament_destroy_fp64, Parament_setHamiltonian_fp64, Parament_equiprop_fp64

.. autocmodule:: cuda/parament.h
   :members: Parament_selectIterationCycles_fp64, Parament_setIterationCyclesManually_fp64, Parament_automaticIterationCycles_fp64, Parament_peekAtLastError_fp64



Constants
~~~~~~~~~

.. autocenum:: cuda/parament.h::Parament_QuadratureSpec
    :members:
    :undoc-members:
    :member-order: bysource

.. autocenum:: cuda/parament.h::Parament_ErrorCode
    :members:
    :undoc-members:
    :member-order: bysource




Python API
----------
.. py:currentmodule:: parament.parament


.. autoclass:: parament.parament.Parament
   :members:
   :undoc-members:

