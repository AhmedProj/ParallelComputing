#include "matrix.h"
#include <iostream>
#include <omp.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

Matrix::Matrix(int n, int m, int p, int q){
    rows = n;
    columns = m;
    depth = p;
    dim4 = q;
    int size = array_size();
    data = new float[size];
    // Initializing the matrix with zeros
    for(int i; i < size; i++){
        data[i] = 0.0;
    } 
} 

Matrix::Matrix() {
}

Matrix::~Matrix() {
    delete[]data;
    data = nullptr;
    }

Matrix::Matrix(const Matrix& matrix){
    rows = matrix.rows;
    columns = matrix.columns;
    depth = matrix.depth;
    dim4 = matrix.dim4;    

    int size = array_size();
    data = new float[size];
    for(int i=0; i < size; i++){
        data[i] = matrix.data[i];
    }
}

Matrix &Matrix::operator=(const Matrix& matrix) {
    if (this != &matrix) {
        // Delete existing data
        delete[] data;

        rows = matrix.rows;
        columns = matrix.columns;
        depth = matrix.depth;
        dim4 = matrix.dim4;

        // Allocate memory for new data and copy
        //cout << " Assign //// " << rows << " " << columns << " " << depth << " " << dim4 << endl;
        data = new float[rows * columns * depth * dim4];
        copy(matrix.data, matrix.data + rows * columns * depth * dim4, data);
    }
    return *this;      
}        

void Matrix::set_value(int i, int j, float value){
    data[i * columns + j] = value;
} 

float Matrix::get_value(int i, int j){
    return data[i * columns + j];
} 

void Matrix::dims_values(){
    dims[0] = rows; dims[1] = columns; dims[2] = depth; dims[3] = dim4; 
}
int Matrix::array_size(){
    int dim = dimension();
    int size = 1; 
    for (int i = 0; i < dim; i++){
        size *= dims[i];
    }
    return size;
}
int Matrix::dimension(){
    int dim = 0;
    dims_values();
    int max_size = sizeof(dims) / sizeof(dims[0]);
    for(int i = 0; i < max_size; i++){
        if (dims[i] != 0){
            dim ++; 
        }
    }
    return dim;   
}

 py::array_t<int> Matrix::shape(){
    int dim = dimension();
    int* sizes = new int[dim];
    for (int i = 0; i < dim; i++){
        sizes[i] = dims[i];
    }
    py::array_t<int> array(dim, sizes);
    delete[] sizes;
    return array;
}

int Matrix::shape_inner(const Matrix& matrix){
    rows = matrix.rows;
    columns = matrix.columns;
    depth = matrix.depth;
    dim4 = matrix.dim4;
    int dim = 0;  
    int max_size = sizeof(dims) / sizeof(dims[0]);
    for(int i = 0; i < max_size; i++){
        if (dims[i] != 0){
            dim ++; 
        }
    }
    return dim;
}


py::array_t<float> Matrix::to_numpy(){
    int dim = dimension();
    int* sizes = new int[dim];
    for (int i = 0; i < dim; i++){
        sizes[i] = dims[i];
    }
    std::vector<pybind11::ssize_t> sizes_vec(sizes, sizes + dim);
    auto array = py::array_t<float>(sizes_vec, data);
    delete[] sizes;
    return array;
}

void Matrix::set_zero(){
    int size = array_size();
    for(int i = 0; i < size; i++){
        data[i] = 0;
    }    
}  

void Matrix::set_random(){
    int size = array_size();
    for(int i = 0; i < size; i++){
        srand(i); // completely random time(NULL) instead i
        data[i] = rand() / double(RAND_MAX);
    }    
}  

void Matrix::set_all(py::array_t<float> values){

    py::buffer_info buf = values.request();
    float* ptr = static_cast<float *>(buf.ptr);
    int dimensions = values.ndim();
    int size = array_size();
    for (int i = 0; i < size; i++){
        data[i] = ptr[i];
    }    
}  

void Matrix::print_matrix(){
    int dim = dimension();
    if (dim > 2){
        throw invalid_argument("The matrix has dimensions bigger than 2");
    } else{
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < columns; j++){
                cout << (data[i * columns + j]) << "\t";
            }
            cout << endl;
        }  
    }      
}          

Matrix Matrix::multiplication(const Matrix& x, const Matrix& y){
    if (x.columns != y.rows) {
        throw invalid_argument("Incompatible matrices for multiplication");
    }

    Matrix results(x.rows, y.columns);
    results.set_zero(); // Added to enforce initialization to zero. Python issue
    for(int i = 0; i < x.rows * y.columns; i++){
        for(int j = 0; j < x.columns; j++){
            results.data[i] += x.data[(i / results.columns) * x.columns + j]*
                y.data[ i % results.columns + j * y.columns];
        }
    }
    return results;
}

Matrix Matrix::parallel_multiplication(const Matrix& x, const Matrix& y){ 
    if (x.columns != y.rows) {
        throw invalid_argument("Incompatible matrices for multiplication");
    }    
    int nProcessors = omp_get_max_threads();
    omp_set_num_threads(nProcessors);
    Matrix results(x.rows, y.columns);
    results.set_zero(); // Added to enforce initialization to zero. Python issue
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

Matrix Matrix::tensor_multiplication(const Matrix& x, const Matrix& y){
    if (x.columns != y.columns || x.rows != y.rows || x.depth != y.depth
        || x.dim4 != y.dim4) {
        throw invalid_argument("Tensors have different sizes");
    }   
    int nProcessors = omp_get_max_threads();
    omp_set_num_threads(nProcessors);
    
    int nsize = x.rows * x.columns * x.depth * x.dim4;
    Matrix results(x.rows, x.columns, x.depth, x.dim4); 
    // BUG TO SOLVE IN THE FUTURE!! 
    // int nsize;
    // Matrix results;
    // if (x.depth != 0 && x.dim4 != 0){
    //     nsize = x.rows * x.columns * x.depth * x.dim4;
    //     results = Matrix(x.rows, x.columns, x.depth, x.dim4);
    // } else{
    //     nsize = x.rows * x.columns * x.depth;
    //     results = Matrix(x.rows, x.columns, x.depth);     
    // }
    //cout << "I am up " << results.rows << " " << results.columns << " " << results.depth << endl;
    results.set_zero(); // Added to enforce initialization to zero. Python issue
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
            temp_result[ID][i] += x.data[i] * y.data[i];               
        }
    }
    // Reducing the results
    for (int i = 0; i < nsize; i++) {
        for (int j = 0; j < nProcessors; j++) {
            results.data[i] += temp_result[j][i];
            //cout << i << "  j  " << j << endl;
        }
    }
    // Free allocated memory
    for (int i = 0; i < nProcessors; i++) {
        delete[] temp_result[i];
    }    
    delete[] temp_result;
    temp_result = nullptr;   
    return results;

}