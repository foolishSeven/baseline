import torch
import numpy as np
from torch.distributions.kl import kl_divergence


def test_permute():
    mat1 = torch.randn(4, 5)
    print(mat1)
    mat2 = mat1.permute(1, 0)  # 转置
    print(mat2)
    mat3 = mat1.permute(-1, -2)

    print(mat3)


def test_load_dataset():
    dataset = np.load('..\\..\\dataset\\amazon\\men.npy', allow_pickle=True).item()
    user_set = dataset['train']
    # torch.save(user_set, 'user_set.pt')  # 用户与商品之间的交互表
    item_set = dataset['feat']
    # torch.save(item_set, 'item_set.pt')  # 商品的特征表
    val_set = dataset['val']
    # torch.save(val_set, 'val_set.pt')  # 用户与商品实际的交互值
    test_set = dataset['test']
    # torch.save(test_set, 'test_set.pt')
    print(test_set)


def test_torch_divergence():
    """
    https://pytorch.org/docs/stable/distributions.html?highlight=kl#torch.distributions.kl.kl_divergence
    :return:
    """
    
    pre_ds = torch.randn(3, 4)
    pre_ds = torch.distributions.Categorical(probs=pre_ds)
    labels = torch.randn(3, 4)
    labels = torch.distributions.Categorical(probs=labels)
    torch_res = kl_divergence(pre_ds, labels)

    print(torch_res)


def test_torch_sum():
    """
      1. torch.sum(input, dim, out=None)
       参数说明：
           input：输入的tensor矩阵。
           dim：求和的方向。若input为2维tensor矩阵，dim=0，对列求和；dim=1，对行求和。注意：输入的形状可为其他维度（3维或4维），可根据dim设定对相应的维度求和。
           out: 输出，一般不指定。
    :return:
    """
    device = torch.device('cuda:0')
    # torch.cuda.init()
    torch.cuda.device(device)
    tensor_a = torch.randn(100, 100)
    tensor_a = tensor_a.cuda()
    tensor_b = torch.randn(100, 100)
    tensor_b = tensor_b.cuda()
    results = torch.mm(tensor_a, tensor_b)
    results_sum = torch.sum(results, dim=1)
    print(results)
    print(results_sum)


if __name__ == '__main__':

    """
    tensor([[-1.6175, -1.4143,  1.5691],
        [ 0.6201,  0.2711, -0.1303]])
    tensor([[ 2.9513, -0.2394],
        [ 0.0209,  0.7037],
        [-1.4697, -0.5638]])
    """
    test_torch_divergence()

