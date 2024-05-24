from ParallelComputing import Matrix


size = 5
A = Matrix(size, size)
B = Matrix(size, size)

A.set_all()
B.set_all()

print(A.get_nrows())
#A.set_value(0, 0,-23.5)
value = A.get_value(0, 0)
print(value,'+++++++++++')

A.print_matrix()

C = Matrix.multiplication(A, B)
CC = Matrix.multiplication(A, B)
C_par = Matrix.parallel_multiplication(A, B)
C_par2 = Matrix.parallel_multiplication(A, B)
C_par.print_matrix()
CC.print_matrix()
print("///////////////////")
C.print_matrix()