cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(gpgpu-arena LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FetchContent)

find_package(CUDAToolkit REQUIRED)

# --- spdlog ---

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.0)
FetchContent_MakeAvailable(spdlog)

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

    FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.90.1)
    FetchContent_MakeAvailable(imgui)

    FetchContent_Declare(implot
        GIT_REPOSITORY https://github.com/epezent/implot.git
        GIT_TAG v0.16)
    FetchContent_MakeAvailable(implot)

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
    src/arena/kernel_compiler.cpp
    src/arena/profiler.cpp
    src/arena/benchmark.cpp
    src/arena/runner.cpp
    src/frontend/cli.cpp
    ${KERNEL_DESCRIPTORS}
    ${CUDA_DESCRIPTORS}
)

if(BUILD_GUI)
    list(APPEND ARENA_SOURCES src/frontend/gui.cpp)
endif()

add_executable(arena ${ARENA_SOURCES})

find_package(Python3 COMPONENTS Interpreter)

target_include_directories(arena PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(arena PRIVATE CUDA::cuda_driver CUDA::cupti CUDA::cudart CUDA::nvtx3 spdlog::spdlog)
target_compile_definitions(arena PRIVATE
    ARENA_KERNEL_DIR="${CMAKE_SOURCE_DIR}/kernels"
    ARENA_PYTHON="${Python3_EXECUTABLE}"
)

if(BUILD_GUI)
    target_link_libraries(arena PRIVATE imgui_lib)
    target_compile_definitions(arena PRIVATE ARENA_GUI_ENABLED)
endif()

# --- PTX compilation ---

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/kernels)
file(GLOB_RECURSE KERNEL_SOURCES "${CMAKE_SOURCE_DIR}/kernels/*/*.cu")

# exclude CUDA descriptors from PTX compilation (they're linked into exe, not loaded as PTX), TODO: clean up this logic
list(FILTER KERNEL_SOURCES EXCLUDE REGEX "cub_.*\\.cu$")
list(FILTER KERNEL_SOURCES EXCLUDE REGEX "thrust_.*\\.cu$")

set(KERNEL_PTXS "")
foreach(KERNEL_SRC ${KERNEL_SOURCES})
    get_filename_component(KERNEL_NAME ${KERNEL_SRC} NAME_WE)
    get_filename_component(KERNEL_DIR ${KERNEL_SRC} DIRECTORY)
    get_filename_component(CATEGORY ${KERNEL_DIR} NAME)

    set(PTX_FILE ${CMAKE_BINARY_DIR}/kernels/${CATEGORY}_${KERNEL_NAME}.ptx)

    add_custom_command(
        OUTPUT ${PTX_FILE}
        COMMAND ${CMAKE_CUDA_COMPILER} -ptx -arch=native -o ${PTX_FILE} ${KERNEL_SRC}
        DEPENDS ${KERNEL_SRC}
        COMMENT "${CATEGORY}/${KERNEL_NAME}.cu -> ptx"
    )
    
    list(APPEND KERNEL_PTXS ${PTX_FILE})
endforeach()

add_custom_target(kernels ALL DEPENDS ${KERNEL_PTXS})
add_dependencies(arena kernels)


# --- info ---

message(STATUS "Kernel descriptors:")
foreach(DESC ${KERNEL_DESCRIPTORS})
    get_filename_component(DESC_NAME ${DESC} NAME)
    message(STATUS "  ${DESC_NAME}")
endforeach()

message(STATUS "Kernel sources:")
foreach(SRC ${KERNEL_SOURCES})
    get_filename_component(SRC_REL ${SRC} NAME)
    message(STATUS "  ${SRC_REL}")
endforeach()

# --- defaults ---

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
