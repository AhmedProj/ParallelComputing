cmake_minimum_required(VERSION 3.12)
project(ParallelComputing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(pybind11)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

pybind11_add_module(ParallelComputing src/matrix.cpp src/matrix_binding.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ParallelComputing PUBLIC OpenMP::OpenMP_CXX "${TORCH_LIBRARIES}")
else()
    message(FATAL_ERROR "OpenMP not found. Please ensure OpenMP is installed and configured properly.")
endif()

target_link_libraries(ParallelComputing PRIVATE ${TORCH_LIBRARIES})





