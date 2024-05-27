#include <pybind11/pybind11.h>
#include "matrix.h"

namespace py = pybind11;

PYBIND11_MODULE(ParallelComputing, m) {
    m.doc() = "Matrix operations module"; 

    py::class_<Matrix>(m, "Matrix")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def(py::init<int, int, int>())
        .def(py::init<int, int, int, int>())
        .def("set_value", &Matrix::set_value)
        .def("get_value", &Matrix::get_value)
        .def("dimension", &Matrix::dimension)
        .def("to_numpy", &Matrix::to_numpy)
        .def("shape", &Matrix::shape)
        .def("set_zero", &Matrix::set_zero)
        .def("set_all", &Matrix::set_all)        
        .def("set_random", &Matrix::set_random)
        .def("print_matrix", &Matrix::print_matrix)
        .def_static("multiplication", &Matrix::multiplication)
        .def_static("parallel_multiplication", &Matrix::parallel_multiplication)
        .def_static("tensor_multiplication", &Matrix::tensor_multiplication);        
}
