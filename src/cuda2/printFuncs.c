/* Copyright 2020 Konstantin Herb. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include "printFuncs.h"


void printcomplex(cuComplex* data, int len) {
    int j = 0;
    for (j = 0; j < len; j++) {
        
        printf("(%5.3f,%5.3fi) ", data[j].x, data[j].y);
    }
    printf("\n");
}
