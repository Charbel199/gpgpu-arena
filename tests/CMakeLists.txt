add_executable(arena_tests
    test_measure_policy.cpp
    test_cli_args.cpp
    ${CMAKE_SOURCE_DIR}/src/arena/measure_policy.cpp
    ${CMAKE_SOURCE_DIR}/src/arena/cli_args.cpp
)

target_include_directories(arena_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(arena_tests PRIVATE doctest::doctest)

add_test(NAME arena_tests COMMAND arena_tests)
