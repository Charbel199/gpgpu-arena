cmake_minimum_required(VERSION 3.18)
project(gpgpu-arena LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# Find CUDA Toolkit (includes Driver API & CUPTI)
# ------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------------------------
# Arena Executable (The "Referee" - your main harness)
# ------------------------------------------------------------------------------
add_executable(arena
    src/main.cpp
    src/arena/context.cpp
    src/arena/kernel_loader.cpp
    src/arena/profiler.cpp
)

target_include_directories(arena PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(arena PRIVATE
    CUDA::cuda_driver    # Driver API (cuModuleLoad, cuLaunchKernel, etc.)
    CUDA::cupti          # CUPTI for profiling metrics
)

# ------------------------------------------------------------------------------
# Gladiator Kernels (compiled to PTX for dynamic loading)
# ------------------------------------------------------------------------------
# Each kernel is compiled separately to .ptx so the arena can load them at runtime

# Ensure kernels directory exists
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/kernels)

set(KERNEL_SOURCES
    ${CMAKE_SOURCE_DIR}/kernels/matmul_naive.cu
    ${CMAKE_SOURCE_DIR}/kernels/matmul_tiled.cu
    # ${CMAKE_SOURCE_DIR}/kernels/matmul_register.cu   # Add as you implement
    # ${CMAKE_SOURCE_DIR}/kernels/matmul_tensor.cu     # Tensor Core version
)

# Detect GPU architecture (default to common ones)
set(CUDA_ARCH "75" CACHE STRING "CUDA architecture to compile for")

set(KERNEL_PTXS "")
foreach(KERNEL_SRC ${KERNEL_SOURCES})
    get_filename_component(KERNEL_NAME ${KERNEL_SRC} NAME_WE)
    set(PTX_FILE ${CMAKE_BINARY_DIR}/kernels/${KERNEL_NAME}.ptx)
    
    # Compile .cu to .ptx using nvcc directly
    add_custom_command(
        OUTPUT ${PTX_FILE}
        COMMAND ${CMAKE_CUDA_COMPILER} 
            -ptx 
            -arch=sm_${CUDA_ARCH}
            -o ${PTX_FILE}
            ${KERNEL_SRC}
        DEPENDS ${KERNEL_SRC}
        COMMENT "Compiling ${KERNEL_NAME}.cu -> ${KERNEL_NAME}.ptx"
    )
    
    list(APPEND KERNEL_PTXS ${PTX_FILE})
endforeach()

# Create a target that builds all kernels
add_custom_target(kernels ALL DEPENDS ${KERNEL_PTXS})

# Make arena depend on kernels
add_dependencies(arena kernels)

# ------------------------------------------------------------------------------
# Build type defaults
# ------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# ------------------------------------------------------------------------------
# Optional: Dear ImGui for the UI (Phase 3)
# ------------------------------------------------------------------------------
option(BUILD_UI "Build with Dear ImGui visualization" OFF)

if(BUILD_UI)
    # You'll add ImGui integration here later
    # find_package(imgui REQUIRED)
    # target_link_libraries(arena PRIVATE imgui::imgui)
    message(STATUS "UI build enabled - ImGui integration pending")
endif()

