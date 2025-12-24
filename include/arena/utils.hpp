#pragma once

#include <cuda.h>
#include <cupti.h>
#include <stdexcept>
#include <string>

namespace arena {


inline void check_cuda(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        throw std::runtime_error(
            std::string(operation) + " failed: " + error_str
        );
    }
}


inline void check_cupti(CUptiResult result, const char* operation) {
    if (result != CUPTI_SUCCESS) {
        const char* error_str;
        cuptiGetResultString(result, &error_str);
        throw std::runtime_error(
            std::string(operation) + " failed: " + error_str
        );
    }
}

}

