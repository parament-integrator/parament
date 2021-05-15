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


# Error codes
PARAMENT_STATUS_SUCCESS = 0
PARAMENT_STATUS_HOST_ALLOC_FAILED = 10
PARAMENT_STATUS_DEVICE_ALLOC_FAILED = 20
PARAMENT_STATUS_CUBLAS_INIT_FAILED = 30
PARAMENT_STATUS_INVALID_VALUE = 50
PARAMENT_STATUS_CUBLAS_FAILED = 60
PARAMENT_STATUS_SELECT_SMALLER_DT = 70
PARAMENT_STATUS_NO_HAMILTONIAN = 80
PARAMENT_STATUS_INVALID_QUADRATURE_SELECTION = 90
PARAMENT_FAIL = 1000

# Quadrature mode
PARAMENT_QUADRATURE_NONE = 0x00000000
PARAMENT_QUADRATURE_MIDPOINT = 0x01000000
PARAMENT_QUADRATURE_SIMPSON = 0x02000000
