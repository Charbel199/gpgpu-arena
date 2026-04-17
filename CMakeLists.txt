cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(gpgpu-arena LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

message(STATUS "")
message(STATUS "=== GPGPU Arena Build Configuration ===")
message(STATUS "")

include(FetchContent)

find_package(CUDAToolkit REQUIRED)

message(STATUS "[CUDA] Toolkit version:  ${CUDAToolkit_VERSION}")
message(STATUS "[CUDA] nvcc:             ${CMAKE_CUDA_COMPILER}")
message(STATUS "[CUDA] Architectures:    ${CMAKE_CUDA_ARCHITECTURES}")

message(STATUS "")
message(STATUS "[deps] Fetching dependencies ...")

# --- spdlog ---

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.0)
FetchContent_MakeAvailable(spdlog)
message(STATUS "[deps]   spdlog v1.15.0")

# --- nlohmann/json (header-only, used by kernel compiler) ---

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3)
FetchContent_MakeAvailable(json)
message(STATUS "[deps]   nlohmann/json v3.11.3")

# --- GUI (ImGui + ImPlot) ---

option(BUILD_GUI "Build with Dear ImGui GUI" ON)

if(BUILD_GUI)
    FetchContent_Declare(glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.3.8)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
    set(CMAKE_FIND_FRAMEWORK NEVER CACHE STRING "" FORCE)
    FetchContent_MakeAvailable(glfw)
    message(STATUS "[deps]   GLFW 3.3.8")

    FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.90.1)
    FetchContent_MakeAvailable(imgui)
    message(STATUS "[deps]   Dear ImGui v1.90.1")

    FetchContent_Declare(implot
        GIT_REPOSITORY https://github.com/epezent/implot.git
        GIT_TAG v0.16)
    FetchContent_MakeAvailable(implot)
    message(STATUS "[deps]   ImPlot v0.16")

    find_package(OpenGL REQUIRED)

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

# --- arena executable ---

file(GLOB_RECURSE KERNEL_DESCRIPTORS "${CMAKE_SOURCE_DIR}/kernels/*/*.cpp")

# CUDA descriptors (use runtime API, e.g. CUB/Thrust) compiled by nvcc, linked into exe
file(GLOB CUDA_DESCRIPTORS "${CMAKE_SOURCE_DIR}/kernels/*/cub_*.cu" "${CMAKE_SOURCE_DIR}/kernels/*/thrust_*.cu")

set(ARENA_SOURCES
    src/main.cpp
    src/arena/context.cpp
    src/arena/kernel_loader.cpp
    src/arena/compilers/compiler_utils.cpp
    src/arena/compilers/cuda_compiler.cpp
    src/arena/compilers/triton_compiler.cpp
    src/arena/compilers/cutile_compiler.cpp
    src/arena/compilers/warp_compiler.cpp
    src/arena/compilers/kernel_compiler.cpp
    src/arena/profiler.cpp
    src/arena/benchmark.cpp
    src/arena/runner.cpp
    src/frontend/tui.cpp
    ${KERNEL_DESCRIPTORS}
    ${CUDA_DESCRIPTORS}
)

if(BUILD_GUI)
    list(APPEND ARENA_SOURCES src/frontend/gui.cpp)
endif()

add_executable(arena ${ARENA_SOURCES})

find_package(Python3 COMPONENTS Interpreter)

if(Python3_FOUND)
    message(STATUS "[Python] ${Python3_EXECUTABLE} (${Python3_VERSION})")
else()
    message(STATUS "[Python] Not found - Triton/cuTile kernels will fail at runtime")
endif()

target_include_directories(arena PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(arena PRIVATE CUDA::cuda_driver CUDA::cupti CUDA::cudart CUDA::nvtx3 spdlog::spdlog nlohmann_json::nlohmann_json)
target_compile_definitions(arena PRIVATE
    ARENA_KERNEL_DIR="${CMAKE_SOURCE_DIR}/kernels"
    ARENA_PYTHON="${Python3_EXECUTABLE}"
    ARENA_NVCC="${CMAKE_CUDA_COMPILER}"
)

if(BUILD_GUI)
    target_link_libraries(arena PRIVATE imgui_lib)
    target_compile_definitions(arena PRIVATE ARENA_GUI_ENABLED)
endif()

# --- defaults ---

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# --- info ---

list(LENGTH KERNEL_DESCRIPTORS NUM_DESCRIPTORS)
list(LENGTH CUDA_DESCRIPTORS NUM_CUDA_DESCRIPTORS)

message(STATUS "")
message(STATUS "[kernels] Found ${NUM_DESCRIPTORS} kernel descriptors + ${NUM_CUDA_DESCRIPTORS} CUDA runtime descriptors:")
foreach(DESC ${KERNEL_DESCRIPTORS})
    get_filename_component(DESC_NAME ${DESC} NAME)
    message(STATUS "[kernels]   ${DESC_NAME}")
endforeach()
foreach(DESC ${CUDA_DESCRIPTORS})
    get_filename_component(DESC_NAME ${DESC} NAME)
    message(STATUS "[kernels]   ${DESC_NAME} (CUDA runtime)")
endforeach()

message(STATUS "")
message(STATUS "=== Build Summary ===")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler:  ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CUDA compiler: ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "  GUI:           ${BUILD_GUI}")
message(STATUS "  Kernel dir:    ${CMAKE_SOURCE_DIR}/kernels")
message(STATUS "")
