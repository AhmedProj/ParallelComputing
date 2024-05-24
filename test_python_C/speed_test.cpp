#include <iostream>
#include <omp.h>
#include <chrono>
#include "matrix.h"

using namespace std;
using namespace std::chrono;

int main()
{
    //for(int size=100; size <= 1000; size+= 100){
    int size = 5;
    Matrix A(size, size);
    Matrix B(size, size);

    A.set_all();  
    B.set_all();

    cout << A.get_nrows() << endl;

    //A.print_matrix();

    //A.print_matrix();
    //cout << "second matrix" << endl; 
    //B.print_matrix();   

    auto start = high_resolution_clock::now();
    Matrix result = Matrix::multiplication(A, B);
    Matrix result2 = Matrix::parallel_multiplication(A, B);
    result.print_matrix();
    result2.print_matrix();
    auto stop = high_resolution_clock::now();

    // Time calculation in microseconds
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "For size " << size << " it took by the function: "
        << duration.count() << " milliseconds" << endl;
    //}     

    return 0;
}