import numpy as np
import torch


def log_cosh_loss(input, target):
    a = input - target
    b = torch.cosh(a)
    loss = torch.log(b)
    return loss.mean()


def mse_side_loss(pred, target):
    # 计算MSE loss
    mse_loss = torch.nn.functional.mse_loss(pred, target)

    # 判断预测值和标签是否在1的同侧
    same_side_loss = torch.where((pred - 1) * (target - 1) < 0, (pred - 1) ** 2, torch.tensor(0.0))

    # 可以根据需要调整mse_loss和same_side_loss的权重
    total_loss = mse_loss + same_side_loss.mean()

    return total_loss


if __name__ == '__main__':
    output = torch.tensor([0.9, 1.1, 1.1, 1.2])

    label = torch.tensor([1.1, 0.9, 1.2, 0.8])
    # loss = log_cosh_loss(output, label)
    loss = mse_side_loss(output, label)
    print(loss)
