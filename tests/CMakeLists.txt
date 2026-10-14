add_executable(arena_tests
    test_measure_policy.cpp
    test_cli_args.cpp
    test_accuracy.cpp
    test_distribution.cpp
    test_sass.cpp
    test_launch_summary.cpp
    test_compiler_cache.cpp
    test_manifest.cpp
    ${CMAKE_SOURCE_DIR}/src/arena/measurement/measure_policy.cpp
    ${CMAKE_SOURCE_DIR}/src/arena/measurement/launch_summary.cpp
    ${CMAKE_SOURCE_DIR}/src/arena/cli_args.cpp
    ${CMAKE_SOURCE_DIR}/src/arena/dtype.cpp
    ${CMAKE_SOURCE_DIR}/src/arena/distribution.cpp
    ${CMAKE_SOURCE_DIR}/src/arena/measurement/accuracy.cpp
    ${CMAKE_SOURCE_DIR}/src/arena/compilers/cache_key.cpp
    ${CMAKE_SOURCE_DIR}/src/arena/compilers/sass.cpp
    ${CMAKE_SOURCE_DIR}/src/arena/compilers/compiler_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/arena/compilers/kernel_compiler.cpp
    ${CMAKE_SOURCE_DIR}/src/arena/manifest/expr.cpp
    ${CMAKE_SOURCE_DIR}/src/arena/manifest/args.cpp
)

# None of these touch the GPU or the CUDA driver, so the tests run on any
# machine, CI runners included. KernelCompiler needs a kernel dir only to
# resolve relative paths; the tests use absolute ones.
target_include_directories(arena_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(arena_tests PRIVATE doctest::doctest spdlog::spdlog nlohmann_json::nlohmann_json)
target_compile_definitions(arena_tests PRIVATE ARENA_KERNEL_DIR="${CMAKE_SOURCE_DIR}/kernels")

add_test(NAME arena_tests COMMAND arena_tests)
