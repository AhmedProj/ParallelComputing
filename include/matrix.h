#include <pybind11/numpy.h>
#pragma once

class Matrix {
private:
    float* data;
    int rows;
    int columns;

public:
    Matrix(int n, int m);
    ~Matrix();
    Matrix(const Matrix& matrix);
    void set_value(int i, int j, float value);
    float get_value(int i, int j);
    int get_nrows();
    int get_ncolumns();
    float* get_data();
    void set_zero();
    void set_all(pybind11::array_t<float> values);    
    void set_random();
    void print_matrix();        
    static Matrix multiplication(const Matrix& x, const Matrix& y);
    static Matrix parallel_multiplication(const Matrix& x, const Matrix& y);
};