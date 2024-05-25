#include <pybind11/pybind11.h>
#include "matrix.h"

namespace py = pybind11;

PYBIND11_MODULE(ParallelComputing, m) {
    m.doc() = "Matrix operations module"; 

    py::class_<Matrix>(m, "Matrix")
        .def(py::init<int, int>())
        .def("set_value", &Matrix::set_value)
        .def("get_value", &Matrix::get_value)
        .def("get_nrows", &Matrix::get_nrows)
        .def("to_numpy", &Matrix::to_numpy)
        .def("get_ncolumns", &Matrix::get_ncolumns)
        .def("set_zero", &Matrix::set_zero)
        .def("set_all", &Matrix::set_all)        
        .def("set_random", &Matrix::set_random)
        .def("print_matrix", &Matrix::print_matrix)
        .def_static("multiplication", &Matrix::multiplication)
        .def_static("parallel_multiplication", &Matrix::parallel_multiplication);
}
