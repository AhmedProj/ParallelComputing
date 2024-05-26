from ParallelComputing import Matrix
from model_llama2 import AttentionModified, ModelArgs
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

    dimension = 128 # 4096 (default)
    args = ModelArgs(dim = dimension)
    attention = AttentionModified(args)

    y_dimension = dimension // 32
    input_tensor = torch.randn(32, y_dimension, dimension)
    start_pos = 0
    freq_cis = torch.randn(y_dimension, y_dimension // 2)

    attention.forward(input_tensor, start_pos, freq_cis, None)


