import multiprocessing
import time
from random import Random

from multiprocessing import Barrier

from multiprocessing.context import Process

from threading import Thread

process_count = 8
matrix_size = 200
random = Random()


def generate_random_matrix_flat(matrix):
    for row in range(matrix_size):
        for col in range(matrix_size):
            matrix[row * matrix_size + col] = random.randint(-5, 5)


def work_out_row_process(id, matrix_a, matrix_b, result, work_start, work_complete):
    while True:
        work_start.wait()
        for row in range(id, matrix_size, process_count):
            for col in range(matrix_size):
                for i in range(matrix_size):
                    result[row * matrix_size + col] += matrix_a[row * matrix_size + i] * matrix_b[i * matrix_size + col]
        work_complete.wait()



def generate_random_matrix_nested(matrix):
    for row in range(matrix_size):
        for col in range(matrix_size):
            matrix[row][col] = random.randint(-5, 5)



def work_out_row_thread(row):
    while True:
        work_start.wait()
        for col in range(matrix_size):
            for i in range(matrix_size):
                result[row][col] += matrix_a[row][i] * matrix_b[i][col]
        work_complete.wait()




if __name__ == '__main__':

    #######################@
    multiprocessing.set_start_method('spawn')
    work_start = Barrier(process_count + 1)
    work_complete = Barrier(process_count + 1)
    matrix_a = multiprocessing.Array('i', [0] * (matrix_size * matrix_size), lock=False)
    matrix_b = multiprocessing.Array('i', [0] * (matrix_size * matrix_size), lock=False)
    result = multiprocessing.Array('i', [0] * (matrix_size * matrix_size), lock=False)
    for p in range(process_count):
        Process(target=work_out_row_process, args=(p, matrix_a, matrix_b, result, work_start, work_complete)).start()
    start = time.time()
    for t in range(10):
        generate_random_matrix_flat(matrix_a)
        generate_random_matrix_flat(matrix_b)
        for i in range(matrix_size * matrix_size):
            result[i] = 0
        work_start.wait()
        work_complete.wait()
    end = time.time()
    print("Done, time taken", end - start)
    #############################

    matrix_a = [[0] * matrix_size for a in range(matrix_size)]
    matrix_b = [[0] * matrix_size for b in range(matrix_size)]
    result = [[0] * matrix_size for r in range(matrix_size)]
    random = Random()

    start = time.time()
    for t in range(10):
        generate_random_matrix_nested(matrix_a)
        generate_random_matrix_nested(matrix_b)
        result = [[0] * matrix_size for r in range(matrix_size)]
        for row in range(matrix_size):
            for col in range(matrix_size):
                for i in range(matrix_size):
                    result[row][col] += matrix_a[row][i] * matrix_b[i][col]
    end = time.time()
    print("Done, time taken", end - start)

    ##############################################
    matrix_a = [[0] * matrix_size for a in range(matrix_size)]
    matrix_b = [[0] * matrix_size for b in range(matrix_size)]
    result = [[0] * matrix_size for r in range(matrix_size)]
    random = Random()

    work_start = Barrier(matrix_size + 1)
    work_complete = Barrier(matrix_size + 1)

    for row in range(matrix_size):
        Thread(target=work_out_row_thread, args=([row])).start()
    start = time.time()
    for t in range(10):
        generate_random_matrix_nested(matrix_a)
        generate_random_matrix_nested(matrix_b)
        result = [[0] * matrix_size for r in range(matrix_size)]
        work_start.wait()
        work_complete.wait()
    end = time.time()
    print("Done, time taken", end - start)

