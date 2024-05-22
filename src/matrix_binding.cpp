#include <pybind11/pybind11.h>
#include "matrix.h"

namespace py = pybind11;

PYBIND11_MODULE(ParallelComputing, m) {
    m.doc() = "Matrix operations module"; 

    py::class_<Matrix>(m, "Matrix")
        .def(py::init<int, int>())
        .def("set_value", &Matrix::set_value)
        .def("get_value", &Matrix::get_value)
        .def("set_all", &Matrix::set_all)
        .def("print_matrix", &Matrix::print_matrix)
        .def_static("multiplication", &Matrix::multiplication)
        .def_static("parallel_multiplication", &Matrix::parallel_multiplication);
}