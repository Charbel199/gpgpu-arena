#include "arena/context.hpp"
#include "arena/utils.hpp"

namespace arena {

Context::Context(int device_id) {
    // Initialize the CUDA Driver API
    check_cuda(cuInit(0), "cuInit");

    // Get device count
    int device_count;
    check_cuda(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
    
    if (device_id >= device_count) {
        throw std::runtime_error(
            "Device " + std::to_string(device_id) + " not found. "
            "Available devices: " + std::to_string(device_count)
        );
    }

    // Get the device
    check_cuda(cuDeviceGet(&device_, device_id), "cuDeviceGet");

    // Get device properties
    char name[256];
    check_cuda(cuDeviceGetName(name, sizeof(name), device_), "cuDeviceGetName");
    device_name_ = name;

    check_cuda(
        cuDeviceGetAttribute(&cc_major_, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_),
        "cuDeviceGetAttribute (major)"
    );
    check_cuda(
        cuDeviceGetAttribute(&cc_minor_, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_),
        "cuDeviceGetAttribute (minor)"
    );

    check_cuda(cuDeviceTotalMem(&total_mem_, device_), "cuDeviceTotalMem");

    // Create a CUDA context
    check_cuda(
        cuCtxCreate(&context_, CU_CTX_SCHED_AUTO, device_),
        "cuCtxCreate"
    );
}

Context::~Context() {
    if (context_) {
        cuCtxDestroy(context_);
    }
}

CUdeviceptr Context::allocate(size_t bytes) {
    CUdeviceptr ptr;
    check_cuda(cuMemAlloc(&ptr, bytes), "cuMemAlloc");
    return ptr;
}

void Context::free(CUdeviceptr ptr) {
    if (ptr) {
        cuMemFree(ptr);
    }
}

void Context::copy_to_device(CUdeviceptr dst, const void* src, size_t bytes) {
    check_cuda(cuMemcpyHtoD(dst, src, bytes), "cuMemcpyHtoD");
}

void Context::copy_to_host(void* dst, CUdeviceptr src, size_t bytes) {
    check_cuda(cuMemcpyDtoH(dst, src, bytes), "cuMemcpyDtoH");
}

}

