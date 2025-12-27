cmake_minimum_required(VERSION 3.18)
project(gpgpu-arena LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FetchContent)

# ------------------------------------------------------------------------------
# Find CUDA Toolkit (includes Driver API & CUPTI)
# ------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------------------------
# Option: Build with GUI (Dear ImGui + ImPlot)
# ------------------------------------------------------------------------------
option(BUILD_GUI "Build with Dear ImGui GUI" ON)

if(BUILD_GUI)
    # GLFW for windowing
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.3.8
    )
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)

    # Dear ImGui
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.90.1
    )
    FetchContent_MakeAvailable(imgui)

    # ImPlot
    FetchContent_Declare(
        implot
        GIT_REPOSITORY https://github.com/epezent/implot.git
        GIT_TAG v0.16
    )
    FetchContent_MakeAvailable(implot)

    # Find OpenGL
    find_package(OpenGL REQUIRED)

    # Build ImGui as a library
    add_library(imgui_lib STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
        ${implot_SOURCE_DIR}/implot.cpp
        ${implot_SOURCE_DIR}/implot_items.cpp
    )
    target_include_directories(imgui_lib PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
        ${implot_SOURCE_DIR}
        ${glfw_SOURCE_DIR}/include
    )
    target_link_libraries(imgui_lib PUBLIC glfw OpenGL::GL)
    target_compile_definitions(imgui_lib PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD=0)
endif()

# ------------------------------------------------------------------------------
# Arena Executable
# ------------------------------------------------------------------------------
set(ARENA_SOURCES
    src/main.cpp
    # Backend
    src/arena/context.cpp
    src/arena/kernel_loader.cpp
    src/arena/profiler.cpp
    src/arena/benchmark.cpp
    # Frontend
    src/frontend/cli.cpp
)

if(BUILD_GUI)
    list(APPEND ARENA_SOURCES
        src/frontend/gui.cpp
    )
endif()

add_executable(arena ${ARENA_SOURCES})

target_include_directories(arena PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(arena PRIVATE
    CUDA::cuda_driver
    CUDA::cupti
)

if(BUILD_GUI)
    target_link_libraries(arena PRIVATE imgui_lib)
    target_compile_definitions(arena PRIVATE ARENA_GUI_ENABLED)
endif()

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

