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
