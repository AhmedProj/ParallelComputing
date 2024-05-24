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
    void set_zero();
    void set_all();
    void print_matrix();        
    static Matrix multiplication(const Matrix& x, const Matrix& y);
    static Matrix parallel_multiplication(const Matrix& x, const Matrix& y);
};