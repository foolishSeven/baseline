import torch
import numpy as np
from math import log
from math import fsum
import sys
import os


def compute_entropy(p_array, base):

    return torch.sum(-p_array * log())


def torch_save():
    tensor_save = torch.randn(100, 100)
    torch.save(tensor_save, 'tensor_save.pt')


if __name__ == '__main__':
    device = torch.device('cuda:0')
    # torch.cuda.init()
    torch.cuda.device(device)
    tensor_a = torch.randn(3, 3)
    tensor_a = tensor_a.cuda()
    tensor_b = torch.randn(3, 3)
    tensor_b = tensor_b.cuda()
    result_sum_row = torch.sum(tensor_a, dim=1)  # 按行加
    results = torch.einsum('ij, ji->i', [tensor_a, tensor_b])
    result = torch.sum(results / result_sum_row)

    print(results)
    print(results / result_sum_row)
    print(result)



