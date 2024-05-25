from ParallelComputing import Matrix
import numpy as np
import random

size = 5
A = Matrix(size, size)
B = Matrix(size, size)
C = Matrix(size, size)
D = Matrix(size, size)

A.set_random()
B.set_random()

# To input data into the class Matrix, the data should be 1D numpy arrays 
random.seed(3)
input_1 = np.random.uniform(low=0, high=1, size=(size, size)).astype(np.float32)
random.seed(7)
input_2 = np.random.uniform(low=0, high=1, size=(size, size)).astype(np.float32)
# Assigning the values to the Matrix object
C.set_all(input_1.flat)
D.set_all(input_2.flat)


result_bench = Matrix.multiplication(C, D)
result_par = Matrix.parallel_multiplication(C, D)
result_np = np.matmul(input_1, input_2)

result_bench.print_matrix()
print("*****************************************************")
result_par.print_matrix()
print("*****************************************************")
print(result_np)
