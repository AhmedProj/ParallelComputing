cmake_minimum_required(VERSION 3.12)
project(ParallelComputing)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(pybind11)

pybind11_add_module(ParallelComputing src/matrix.cpp src/matrix_binding.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ParallelComputing PUBLIC OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "OpenMP not found. Please ensure OpenMP is installed and configured properly.")
endif()





