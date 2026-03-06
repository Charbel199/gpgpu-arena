cmake_minimum_required(VERSION 3.18)
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
    src/arena/profiler.cpp
    src/arena/benchmark.cpp
    src/frontend/cli.cpp
    ${KERNEL_DESCRIPTORS}
    ${CUDA_DESCRIPTORS}
)

if(BUILD_GUI)
    list(APPEND ARENA_SOURCES src/frontend/gui.cpp)
endif()

add_executable(arena ${ARENA_SOURCES})

target_include_directories(arena PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(arena PRIVATE CUDA::cuda_driver CUDA::cupti CUDA::cudart spdlog::spdlog)

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

set(CUDA_ARCH "75" CACHE STRING "CUDA architecture to compile for")

set(KERNEL_PTXS "")
foreach(KERNEL_SRC ${KERNEL_SOURCES})
    get_filename_component(KERNEL_NAME ${KERNEL_SRC} NAME_WE)
    get_filename_component(KERNEL_DIR ${KERNEL_SRC} DIRECTORY)
    get_filename_component(CATEGORY ${KERNEL_DIR} NAME)
    
    set(PTX_FILE ${CMAKE_BINARY_DIR}/kernels/${CATEGORY}_${KERNEL_NAME}.ptx)
    
    add_custom_command(
        OUTPUT ${PTX_FILE}
        COMMAND ${CMAKE_CUDA_COMPILER} -ptx -arch=sm_${CUDA_ARCH} -o ${PTX_FILE} ${KERNEL_SRC} #TODO: Possibly at -G for debug builds
        DEPENDS ${KERNEL_SRC}
        COMMENT "${CATEGORY}/${KERNEL_NAME}.cu -> ptx"
    )
    
    list(APPEND KERNEL_PTXS ${PTX_FILE})
endforeach()

add_custom_target(kernels ALL DEPENDS ${KERNEL_PTXS})
add_dependencies(arena kernels)


find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import triton; print(triton.__version__)"
        RESULT_VARIABLE TRITON_CHECK_RESULT
        OUTPUT_VARIABLE TRITON_VERSION
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(TRITON_CHECK_RESULT EQUAL 0)
        message(STATUS "Triton found: ${TRITON_VERSION}")

        # always have .triton.py extension for triton kernels to be compiled
        file(GLOB TRITON_KERNELS "${CMAKE_SOURCE_DIR}/kernels/*/*.triton.py")

        set(TRITON_PTXS "")
        foreach(TRITON_SRC ${TRITON_KERNELS})
            get_filename_component(KERNEL_FULL_NAME ${TRITON_SRC} NAME_WE)  # e.g., "reduce.triton"
            string(REPLACE ".triton" "" KERNEL_NAME ${KERNEL_FULL_NAME})     # e.g., "reduce"
            get_filename_component(KERNEL_DIR ${TRITON_SRC} DIRECTORY)
            get_filename_component(CATEGORY ${KERNEL_DIR} NAME)

            set(PTX_FILE ${CMAKE_BINARY_DIR}/kernels/${CATEGORY}_triton_${KERNEL_NAME}.ptx)

            add_custom_command(
                OUTPUT ${PTX_FILE}
                COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_SOURCE_DIR}/kernels
                        ${Python3_EXECUTABLE} ${TRITON_SRC}
                        --output-dir ${CMAKE_BINARY_DIR}/kernels
                        --output-name ${CATEGORY}_triton_${KERNEL_NAME}
                DEPENDS ${TRITON_SRC} ${CMAKE_SOURCE_DIR}/kernels/triton_base.py
                COMMENT "Triton: ${CATEGORY}/${KERNEL_FULL_NAME}.py -> ptx"
            )

            list(APPEND TRITON_PTXS ${PTX_FILE})
        endforeach()

        if(TRITON_PTXS)
            add_custom_target(triton_kernels ALL DEPENDS ${TRITON_PTXS})
            add_dependencies(arena triton_kernels)

            message(STATUS "Triton kernels:")
            foreach(SRC ${TRITON_KERNELS})
                get_filename_component(SRC_NAME ${SRC} NAME)
                message(STATUS "  ${SRC_NAME}")
            endforeach()
        endif()
    else()
        message(STATUS "Triton not found - Triton kernels will be skipped")
        message(STATUS "Install with: pip install triton torch")
    endif()
endif()

# --- info --- #TODO: Compile kernels on runtime

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
