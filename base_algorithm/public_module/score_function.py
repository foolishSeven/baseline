import torch
import numpy as np


def auc_calculate(labels, pre_ds):
    print('.auc')
    labels = torch.from_numpy(labels)
    labels = labels.cuda()
    result_tensor = torch.Tensor(len(labels))
    pre_ds = torch.from_numpy(pre_ds)
    pre_ds = pre_ds.cuda()
    # 计算tp，阈值设置为0.0，即预测集中，值大于0的样例设置为1，小于0的样例 设置为0
    one = torch.ones_like(pre_ds)
    zero = torch.zeros_like(pre_ds)
    boundary = torch.mean(pre_ds)
    pre_ds = torch.where(pre_ds > boundary, one, zero)
    positive_lens = torch.sum(labels, dim=1)
    negative_lens = torch.abs(torch.sum(labels - 1, dim=1))
    positive_samples = torch.einsum('ij, ij->i', [labels, pre_ds])
    negative_samples = torch.einsum('ij, ij->i', [torch.abs[labels - 1], pre_ds])
    positive_results = positive_samples / positive_lens
    negative_results = negative_samples / negative_lens
    results = torch.mean((negative_results + positive_results) / 2.0)
    return results

    # for i in range(len(labels)):
    #     label = labels[i]
    #     pre_d = pre_ds[i]
    #
    #     positive_len = sum(label)
    #     negative_len = len(label) - positive_len
    #
    #     tp = torch.sum(label * pre_d) / float(positive_len)
    #     fp = torch.abs(torch.sum((label - 1) * pre_d)) / float(negative_len)
    #
    #     result = torch.mean(torch.stack([tp.data, fp.data]), 0)
    #     result_tensor[i] = result
    # return np.array(torch.mean(result_tensor, 0))