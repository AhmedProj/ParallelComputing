from ParallelComputing import Matrix
import numpy as np
import torch
import random

size = 5
# Matrix declaration
C = Matrix(size, size)
D = Matrix(size, size)

# Data input as np.float32 numpy arrays
random.seed(3)
input_1 = np.random.uniform(low=0, high=1, size=(size, size)).astype(np.float32)
random.seed(7)
input_2 = np.random.uniform(low=0, high=1, size=(size, size)).astype(np.float32)
# Assigning the values to the Matrix object
C.set_all(input_1)
D.set_all(input_2)

result_bench = Matrix.multiplication(C, D)
result_par = Matrix.parallel_multiplication(C, D)
result_np = np.matmul(input_1, input_2)

print('-------------- Matrix multiplication without parallelization ---------------')
result_bench.print_matrix()
print('-------------- Matrix multiplication parallelized ---------------')
result_par.print_matrix()
print('-------------- Matrix multiplication using numpy ---------------')
print(result_np)

print("\n*****************************************************")
print("********************* TENSORS ***********************")
print("*****************************************************\n")
# Tensors
# Using Pytorch
A = torch.randn(4, 2, 4, 3)
B = torch.randn(4, 2, 4, 3)
C = torch.multiply(A, B)
print('-------------- Tensor multiplication using pytorch ---------------')
print(C, C.shape)
# Using our class Matrix
m1 = Matrix(*A.shape)
m2 = Matrix(*B.shape)
m1.set_all(A.numpy())
m2.set_all(B.numpy())
mine = Matrix.tensor_multiplication(m1, m2)
mine = mine.to_numpy() # Transformation to numpy
print('-------------- Tensor multiplication using parallelized multiplication ---------------')
print(mine, mine.shape)