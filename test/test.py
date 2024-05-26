from ParallelComputing import Matrix
import numpy as np
import torch
import random

size = 5
A = Matrix(size, size)
B = Matrix(size, size)
C = Matrix(size, size)
D = Matrix(size, size)

A.set_random()
B.set_zero()


# To input data into the class Matrix, the data should be 1D numpy arrays 
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
value = result_par.to_numpy().reshape(size, size)
print(value, type(value))
print('-------------------------------------------')

result_bench.print_matrix()
print("*****************************************************")
result_par.print_matrix()
print("*****************************************************")
print(result_np)
print(result_par.dimension(), result_par.shape())

# Tensors

A = torch.randn(4, 2, 4, 3)
B = torch.randn(4, 2, 4, 3)
C = torch.multiply(A, B)
print(C, C.shape)

print("#################################")
# reshaped_A = A.reshape(-1, A.shape[-1])
# reshaped_B = B.reshape(-1, B.shape[-1])
# m1 = Matrix(*reshaped_A.shape)
# m2 = Matrix(*reshaped_B.shape)
# m1.set_all(torch.flatten(reshaped_A).numpy())
# m2.set_all(torch.flatten(reshaped_B).numpy())

m1 = Matrix(*A.shape)
m2 = Matrix(*B.shape)
m1.set_all(A.numpy())
m2.set_all(B.numpy())
print("I get up to here!")
mine = Matrix.tensor_multiplication(m1, m2)
mine = mine.to_numpy()
print(mine, mine.shape)