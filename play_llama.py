from ParallelComputing import Matrix
from model_llama2 import AttentionModified, Attention, ModelArgs
import torch
import torch.distributed as dist
from fairscale.nn.model_parallel import initialize_model_parallel
import os
import time

def init_distributed():
    dist.init_process_group(backend='nccl')

def setup_model_parallel():
    initialize_model_parallel(model_parallel_size_ = 1)


if __name__ == "__main__":
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23456'

    init_distributed()

    setup_model_parallel()
    t_original, t_modified = [], []
    # Getting the right dimensions
    start = 128
    end = 8000
    dimensions = []
    k = start
    while k <= end:
        dimensions.append(k)
        k *= 2
    for dimension in dimensions:
        print("Matrix size ", dimension)
        args = ModelArgs(dim = dimension)
        attention = AttentionModified(args)
        original_attention = Attention(args)

        # Initializing parameters
        y_dimension = dimension // 32
        input_tensor = torch.randn(32, y_dimension, dimension)
        start_pos = 0
        freq_cis = torch.randn(y_dimension, y_dimension // 2)
        # Time measurement
        time0 = time.time()
        attention.forward(input_tensor, start_pos, freq_cis, None)
        time1 = time.time()
        t_modified.append(round((time1 - time0) * 1000, 3))

        original_attention.forward(input_tensor, start_pos, freq_cis, None)
        time2 = time.time()
        t_original.append(round((time2 - time1) * 1000, 3)) 
        print(t_modified, t_original)
        torch.cuda.empty_cache()

    with open('time_comparison.txt', 'w') as file:
        file.write("# Matrix_size \t t_modified \t t_original \n")
        for item0, item1, item2 in zip(dimensions, t_modified, t_original):
            file.write(f"{item0}\t{item1}\t{item2}\n")
