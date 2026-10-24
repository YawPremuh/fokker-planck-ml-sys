cmake_minimum_required(VERSION 3.16)

project(
    FluxFP
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(
    fluxfp_solver
    src/solver.cpp
)

target_include_directories(
    fluxfp_solver
    PUBLIC include
)

target_compile_options(
    fluxfp_solver
    PRIVATE -Wall -Wextra -Wpedantic
)

add_executable(
    fluxfp
    src/main.cpp
)

target_link_libraries(
    fluxfp
    PRIVATE fluxfp_solver
)

enable_testing()

add_executable(
    solver_tests
    tests/solver_tests.cpp
)

target_link_libraries(
    solver_tests
    PRIVATE fluxfp_solver
)

add_test(
    NAME analytical_tests
    COMMAND solver_tests
)