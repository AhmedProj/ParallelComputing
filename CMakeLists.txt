cmake_minimum_required(VERSION 3.4)
project(parallel_multiplication)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(pybind11)
find_package(OpenMP REQUIRED)

add_library(parallel_multiplication MODULE main.cpp matrix.cpp)
target_link_libraries(parallel_multiplication PRIVATE pybind11::module OpenMP::OpenMP_CXX)