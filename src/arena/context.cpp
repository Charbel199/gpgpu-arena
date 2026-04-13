#include "arena/context.hpp"
#include "arena/utils.hpp"
#include <spdlog/spdlog.h>
#include <cuda.h>

namespace arena {

Context::Context(int device_id) {
    check_cuda(cuInit(0), "cuInit");

    int device_count;
    check_cuda(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
    
    if (device_id >= device_count) {
        throw std::runtime_error(
            "Device " + std::to_string(device_id) + " not found. "
            "Available devices: " + std::to_string(device_count)
        );
    }

    check_cuda(cuDeviceGet(&device_, device_id), "cuDeviceGet");

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

    check_cuda(
        cuDeviceGetAttribute(&sm_count_, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device_),
        "cuDeviceGetAttribute (sm_count)"
    );

    check_cuda(
        cuDeviceGetAttribute(&clock_rate_khz_, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device_),
        "cuDeviceGetAttribute (clock_rate)");
    check_cuda(
        cuDeviceGetAttribute(&memory_clock_khz_, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device_),
        "cuDeviceGetAttribute (mem_clock)");
    check_cuda(
        cuDeviceGetAttribute(&memory_bus_width_, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device_),
        "cuDeviceGetAttribute (bus_width)");

    check_cuda(cuDeviceTotalMem(&total_mem_, device_), "cuDeviceTotalMem");

    auto log = spdlog::get("context");
    log->info("Device #{}: {}", device_id, device_name_);
    log->info("  Compute capability: sm_{}{}", cc_major_, cc_minor_);
    log->info("  SMs: {}", sm_count_);
    log->info("  Memory: {} MB", total_mem_ / (1024 * 1024));

    // cuCtxCreate API changed in CUDA 13.0
#if CUDA_VERSION >= 13000
    CUctxCreateParams params = {};
    check_cuda(
        cuCtxCreate_v4(&context_, &params, CU_CTX_SCHED_AUTO, device_),
        "cuCtxCreate"
    );
#else
    check_cuda(
        cuCtxCreate(&context_, CU_CTX_SCHED_AUTO, device_),
        "cuCtxCreate"
    );
#endif
}

Context::~Context() {
    if (context_) {
        cuCtxDestroy(context_);
    }
}

void Context::reset() {
    auto log = spdlog::get("context");
    log->warn("Resetting CUDA context (recovering from device error) ...");

    if (context_) {
        cuCtxDestroy(context_);
        context_ = nullptr;
    }

    // clear device-level error state
    cuDevicePrimaryCtxReset(device_);

    CUresult result;
#if CUDA_VERSION >= 13000
    CUctxCreateParams params = {};
    result = cuCtxCreate_v4(&context_, &params, CU_CTX_SCHED_AUTO, device_);
#else
    result = cuCtxCreate(&context_, CU_CTX_SCHED_AUTO, device_);
#endif

    if (result != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(result, &err_str);
        log->error("Context reset failed: {} - subsequent kernels will fail", err_str ? err_str : "unknown");
        context_ = nullptr;
    }
}

CUdeviceptr Context::allocate(size_t bytes) {
    CUdeviceptr ptr;
    check_cuda(cuMemAlloc(&ptr, bytes), "cuMemAlloc");
    return ptr;
}

void Context::free(CUdeviceptr ptr) {
    if (ptr) {
        check_cuda(cuMemFree(ptr), "cuMemFree");
    }
}

void Context::copy_to_device(CUdeviceptr dst, const void* src, size_t bytes) {
    check_cuda(cuMemcpyHtoD(dst, src, bytes), "cuMemcpyHtoD");
}

void Context::copy_to_host(void* dst, CUdeviceptr src, size_t bytes) {
    check_cuda(cuMemcpyDtoH(dst, src, bytes), "cuMemcpyDtoH");
}

}

