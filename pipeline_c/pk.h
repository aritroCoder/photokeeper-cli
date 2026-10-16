// C ABI for app shells (Flutter dart:ffi and the like). Everything else in pipeline_c/ is C++.
#pragma once
#include <stddef.h>

#ifdef _WIN32
#define PK_API __declspec(dllexport)
#else
#define PK_API __attribute__((visibility("default")))
#endif
#define PK_N 5  // == pk::N, features in pk::SIGNAL_NAMES order

#ifdef __cplusplus
extern "C" {
#endif

// Directory holding the two ONNX models. Call before the first analyze; the default is the build-time models/ dir.
PK_API void pk_init(const char* models_dir);
// Fill out[] with the features; a signal that failed is NaN. Return 0, or -1 if the image cannot be decoded.
PK_API int pk_analyze_path(const char* path, double out[PK_N]);
PK_API int pk_analyze_encoded(const unsigned char* data, size_t len, double out[PK_N]);  // JPEG/PNG/... bytes in memory
PK_API int pk_keep(const double features[PK_N]);  // 1 keep, 0 delete

#ifdef __cplusplus
}
#endif
