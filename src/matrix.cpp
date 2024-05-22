#include "matrix.h"
#include <iostream>
#include <omp.h>

using namespace std;

Matrix::Matrix(int n, int m){
    rows = n;
    columns = m;
    data = new float[ rows * columns];
    // Initializing the matrix with zeros
    for(int i; i < rows * columns; i++){
        data[i] = 0.0;
    } 
} 

Matrix::~Matrix() {
    delete[]data;
    data = nullptr;
    }

Matrix::Matrix(const Matrix& matrix){
    rows = matrix.rows;
    columns = matrix.columns;
    data = new float[ rows * columns];
    for(int i=0; i < rows * columns; i++){
        data[i] = matrix.data[i];
    }
}

void Matrix::set_value(int i, int j, float value){
    data[i * columns + j] = value;
} 

float Matrix::get_value(int i, int j){
    return data[i * columns + j];
}  

void Matrix::set_all(){
    fill(data, data + rows * columns, 2.378);
}  

void Matrix::print_matrix(){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            cout << (data[i * columns + j]) << "\t";
        }
        cout << endl;
    }    
}          

Matrix Matrix::multiplication(const Matrix& x, const Matrix& y){
    if (x.columns != y.rows) {
    throw invalid_argument("Incompatible matrices for multiplication");
    }

    Matrix results(x.rows, y.columns);
    for(int i = 0; i < x.rows * y.columns; i++){
        for(int j = 0; j < x.columns; j++){
            results.data[i] += x.data[(i / results.columns) * x.columns + j]*
                y.data[ i % results.columns + j * y.columns];
        }
    }
    return results;
}

Matrix Matrix::parallel_multiplication(const Matrix& x, const Matrix& y){
    int nProcessors = omp_get_max_threads();
    omp_set_num_threads(nProcessors);
    Matrix results(x.rows, y.columns);
    int nsize = x.rows * y.columns;
    // Temporary arrays for each thread
    float** temp_result = new float*[nProcessors];
    for (int i = 0; i < nProcessors; i++) {
        temp_result[i] = new float[nsize]();
    }

    #pragma omp parallel num_threads(nProcessors)
    {
        int ID = omp_get_thread_num();
        int block_start = ID * nsize / nProcessors;
        int block_end = (ID + 1) * nsize / nProcessors;

        for(int i = block_start; i < block_end; i++){
            for(int j = 0; j < x.columns; j++){
                
                temp_result[ID][i] += x.data[(i / results.columns) * x.columns + j]*
                    y.data[ i % results.columns + j * y.columns];               
            }
            //cout << "Blocks \t" << block_start << "\t" << ID << endl;
        }
    }    
    // Reducing the results
    for (int i = 0; i < nsize; i++) {
        for (int j = 0; j < nProcessors; j++) {
            results.data[i] += temp_result[j][i];
            //cout << "rank " << j << "\t" << temp_result[j][i] << endl;    
        }
    }
    // Free allocated memory
    for (int i = 0; i < nProcessors; i++) {
        delete[] temp_result[i];
    }    
    delete[] temp_result;

    return results;
}