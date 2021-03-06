/* Copyright 2021 Konstantin Herb, Pol Welter. All Rights Reserved.

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


#ifndef DEVICEINFO_H_
#define DEVICEINFO_H_

#if defined(_MSC_VER)
    // Microsoft
    #define LIBSPEC __declspec(dllexport)
#elif defined(__GNUC__)
    // GCC 
    #define LIBSPEC __attribute__((visibility("default")))
#else
    // do nothing and hope for the best?
    #define LIBSPEC
#endif


#ifdef __cplusplus
extern "C" {
#endif

LIBSPEC void device_info(void);

#ifdef __cplusplus
}
#endif

#endif
