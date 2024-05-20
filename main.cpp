#include <iostream>
#include <omp.h>
#include "matrix.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(parallel_multiplication, m) {
    m.doc() = "pybind11 parallel multiplication"; // optional module docstring
    py::class_<Matrix>(m, "MatrixMultiplication")
        .def(py::init<>())
        .def("multiply", &Matrix::parallel_multiplication);
}