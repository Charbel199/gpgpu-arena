#include <iostream>
#include <vector>
#include <iomanip>

#include "arena/context.hpp"
#include "arena/kernel_loader.hpp"
#include "arena/profiler.hpp"

void print_banner() {
    std::cout << R"(
   ╔═══════════════════════════════════════════════════════════╗
   ║   ⚔️  GPGPU ARENA - Kernel Benchmarking Platform  ⚔️        ║
   ╚═══════════════════════════════════════════════════════════╝
)" << std::endl;
}

void print_device_info(const arena::Context& ctx) {
    std::cout << "🖥️  GPU Device: " << ctx.device_name() << "\n";
    std::cout << "   Compute Capability: " << ctx.compute_capability_major() 
              << "." << ctx.compute_capability_minor() << "\n";
    std::cout << "   Total Memory: " 
              << (ctx.total_memory() / (1024 * 1024)) << " MB\n";
    std::cout << std::string(60, '-') << "\n\n";
}

int main(int argc, char** argv) {
    print_banner();

    try {
        // Phase 1: Initialize the Arena Context
        std::cout << "🏟️  Initializing Arena...\n\n";
        arena::Context ctx(0);  // Use GPU 0
        print_device_info(ctx);

        arena::KernelLoader loader;
        arena::Profiler profiler;

        // Matrix dimensions for testing
        const int M = 1024;  // Rows of A
        const int K = 1024;  // Cols of A / Rows of B  
        const int N = 1024;  // Cols of B

        // Allocate matrices on GPU
        size_t size_a = M * K * sizeof(float);
        size_t size_b = K * N * sizeof(float);
        size_t size_c = M * N * sizeof(float);

        CUdeviceptr d_a = ctx.allocate(size_a);
        CUdeviceptr d_b = ctx.allocate(size_b);
        CUdeviceptr d_c = ctx.allocate(size_c);

        // Initialize host data
        std::vector<float> h_a(M * K, 1.0f);
        std::vector<float> h_b(K * N, 1.0f);
        std::vector<float> h_c(M * N, 0.0f);

        // Copy to device
        ctx.copy_to_device(d_a, h_a.data(), size_a);
        ctx.copy_to_device(d_b, h_b.data(), size_b);

        std::cout << "📊 Matrix Multiplication Benchmark (" 
                  << M << "x" << K << " × " << K << "x" << N << ")\n";
        std::cout << std::string(60, '-') << "\n\n";

        // Load kernels from PTX files
        std::vector<std::string> kernel_files = {
            "kernels/matmul_naive.ptx",
            "kernels/matmul_tiled.ptx"
        };

        for (const auto& kernel_file : kernel_files) {
            try {
                std::cout << "⚔️  Loading Gladiator: " << kernel_file << "\n";
                
                CUmodule module = loader.load_module(kernel_file);
                CUfunction func = loader.get_function(module, "matmul");

                // Configure launch
                arena::KernelLoader::LaunchConfig config;
                config.block_x = 16;
                config.block_y = 16;
                config.grid_x = (N + config.block_x - 1) / config.block_x;
                config.grid_y = (M + config.block_y - 1) / config.block_y;

                // Prepare kernel arguments
                void* args[] = { &d_a, &d_b, &d_c, (void*)&M, (void*)&K, (void*)&N };

                // Warmup runs
                for (int i = 0; i < 10; i++) {
                    loader.launch(func, config, args);
                }

                // Profile the kernel launch
                auto metrics = profiler.profile([&]() { // Creates a lambda function from the .launch() call
                    loader.launch(func, config, args);
                });

                // Calculate GFLOPS
                double flops = 2.0 * M * N * K;
                double gflops = (flops / (metrics.elapsed_ms / 1000.0)) / 1e9;

                std::cout << "   ✅ Time: " << std::fixed << std::setprecision(3) 
                          << metrics.elapsed_ms << " ms\n";
                std::cout << "   📈 Performance: " << std::setprecision(2) 
                          << gflops << " GFLOPS\n";
                
                // Show CUPTI metrics (when available)
                if (metrics.achieved_occupancy > 0) {
                    std::cout << "   🎯 Occupancy: " << std::setprecision(1) 
                              << (metrics.achieved_occupancy * 100) << "%\n";
                }
                if (metrics.dram_read_throughput_gbps > 0) {
                    std::cout << "   💾 DRAM Read: " << std::setprecision(1) 
                              << metrics.dram_read_throughput_gbps << " GB/s\n";
                    std::cout << "   💾 DRAM Write: " << std::setprecision(1) 
                              << metrics.dram_write_throughput_gbps << " GB/s\n";
                }
                std::cout << "\n";

            } catch (const std::exception& e) {
                std::cout << "   ⚠️  Skipped (not compiled yet): " << e.what() << "\n\n";
            }
        }

        // Cleanup
        ctx.free(d_a);
        ctx.free(d_b);
        ctx.free(d_c);

        std::cout << "🏆 Arena session complete!\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

