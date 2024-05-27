#include <pybind11/numpy.h>
#pragma once

class Matrix {
private:
    float* data;
    int rows;
    int columns;
    int depth;
    int dim4;
    int dims[4];
    int shape_inner(const Matrix& matrix); 

public:
    Matrix();
    Matrix& operator=(const Matrix& matrix);
    Matrix(int n, int m, int p = 0, int q = 0);
    ~Matrix();
    Matrix(const Matrix& matrix);
    void set_value(int i, int j, float value);
    float get_value(int i, int j);
    void dims_values();
    int array_size();
    int dimension();
    pybind11::array_t<int> shape();
    pybind11::array_t<float> to_numpy();
    void set_zero();
    void set_all(pybind11::array_t<float> values);    
    void set_random();
    void print_matrix();        
    static Matrix multiplication(const Matrix& x, const Matrix& y);
    static Matrix parallel_multiplication(const Matrix& x, const Matrix& y);
    static Matrix tensor_multiplication(const Matrix& x, const Matrix& y);

};