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

import parament
import pytest
import numpy as np


@pytest.fixture
def context():
    with parament.Parament() as runner:
        yield runner


def test_use_after_destruction():
    context = parament.Parament()
    context.destroy()
    with pytest.raises(RuntimeError, match='Attempting to use a context that has been destroyed'):
        context.set_hamiltonian(np.eye(2), np.eye(2))


def test_propagate_no_hamiltonian(context: parament.Parament):
    with pytest.raises(RuntimeError, match='No hamiltonian set'):
        context.equiprop(1.0)


def test_invalid_quadrature_selection(context: parament.Parament):
    with pytest.raises(ValueError, match='Invalid quadrature selection'):
        context.set_hamiltonian(np.eye(2), np.eye(2), use_magnus=True, quadrature_mode='none')
    with pytest.raises(ValueError, match='Invalid quadrature selection'):
        context.set_hamiltonian(np.eye(2), np.eye(2), use_magnus=True, quadrature_mode='midpoint')


def test_propagate_unset_amplitudes(context: parament.Parament):
    context.set_hamiltonian(np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match='Got 2 amplitude arrays, but there are only 1 Hamiltonians.'):
        context.equiprop(1.0, np.zeros(4), np.zeros(4))


def test_propagate_unequal_amplitude_size(context: parament.Parament):
    context.set_hamiltonian(np.eye(2), np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match='All amplitude arrays must have the same length.'):
        context.equiprop(1.0, np.zeros(4), np.zeros(5))
