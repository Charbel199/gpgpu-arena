#include "arena/categories/matmul_base.hpp"
#include <cublas_v2.h>

namespace arena {

// cuBLAS SGEMM as the reference the hand-written matmuls are measured
// against. Plain FP32: CUBLAS_PEDANTIC_MATH keeps TF32 and tensor cores out,
// so it does the same arithmetic the other fp32 kernels are held to.
class MatmulCublasDescriptor : public MatmulDescriptorBase {
public:
    std::string name() const override { return "matmul_cublas"; }
    std::string description() const override {
        return "cuBLAS SGEMM reference (FP32, pedantic math, no TF32)";
    }
    std::string module_path() const override { return ""; }
    bool uses_module() const override { return false; }
    bool reference() const override { return true; }

    void allocate(Context& ctx) override {
        MatmulDescriptorBase::allocate(ctx);

        // One handle per run, created against the context that is current
        // now. The runner can recreate its context between runs, so a handle
        // kept across runs could point at a dead one.
        check(cublasCreate(&handle_), "cublasCreate");
        // Stream 0, the legacy default stream: the runner's timing events and
        // every other launch go there, so the events bracket this call too.
        check(cublasSetStream(handle_, nullptr), "cublasSetStream");
        check(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
        check(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
    }

    // A, B and C are row-major. cuBLAS is column-major, and a row-major
    // matrix read as column-major is its transpose, so this asks for
    // C^T = B^T * A^T: an N x M result from B^T (N x K, ld N) and
    // A^T (K x M, ld K), written to C with ld N. No transposes are needed.
    void execute(Context&) override {
        const float alpha = 1.0f, beta = 0.0f;
        check(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                          N_, M_, K_,
                          &alpha,
                          reinterpret_cast<const float*>(d_b_), N_,
                          reinterpret_cast<const float*>(d_a_), K_,
                          &beta,
                          reinterpret_cast<float*>(d_c_), N_),
              "cublasSgemm");
    }

    void cleanup(Context& ctx) override {
        if (handle_) cublasDestroy(handle_);
        handle_ = nullptr;
        MatmulDescriptorBase::cleanup(ctx);
    }

    KernelLoader::LaunchConfig get_launch_config() const override { return {}; }

private:
    cublasHandle_t handle_ = nullptr;

    static void check(cublasStatus_t s, const char* what) {
        if (s != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error(std::string(what) + " failed: " + cublasGetStatusString(s));
    }
};

REGISTER_KERNEL(MatmulCublasDescriptor);

}
