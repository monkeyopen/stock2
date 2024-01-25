import random

from dotenv import load_dotenv
import os

from torch.optim.lr_scheduler import StepLR

load_dotenv()
CONF_PATH = os.getenv('CONF_PATH')
DATA_PATH = os.getenv('DATA_PATH')
ROOT_PATH = os.getenv('ROOT_PATH')
import sys

# print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([ROOT_PATH])

import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Sampler, RandomSampler
from get_fortune import Backtesting
from mynet.neural_network import FiveLayerNN
from collections import Counter


class CustomSampler(Sampler):
    def __init__(self, dataset, neg_sample_prob=0.5):
        self.dataset = dataset
        self.neg_sample_prob = neg_sample_prob
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        random.shuffle(self.indices)  # 随机打乱索引
        for idx, (data, label) in enumerate(self.dataset):
            if label == 0:  # 如果是负样本，按照指定的概率抽取
                if random.random() < self.neg_sample_prob:
                    yield idx
            else:  # 如果是正样本，全部抽取
                yield idx

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    features_list = []
    labels_list = []
    infos_list = []
    label_file = CONF_PATH + "us_stock_test"
    model_name = "model/buy_weights_sample_20240125"
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    with open(label_file, 'r') as f:
        for line in f.readlines():
            stock = line.strip()
            if len(stock) < 1:
                continue
            stock_path = DATA_PATH + stock
            backtest = Backtesting(
                data_dir=stock_path,
                dt_format='%Y-%m-%d',
                start_date=datetime.datetime(2000, 1, 1),
                end_date=datetime.datetime(2024, 1, 31),
                sample_start=datetime.datetime(2016, 1, 1),
                sample_end=datetime.datetime(2023, 12, 1)
            )
            backtest._read_data()
            feature, label, info = backtest.generate_samples(buy=1)
            if len(feature) > 0:
                features_list.append(feature)
                labels_list.append(label)
                infos_list.append(info)
    labels = np.concatenate(labels_list)
    label_counter = Counter(labels)
    print("Label distribution:")
    print(f"Positive samples (1): {label_counter[1]}")
    print(f"Negative samples (0): {label_counter[0]}")
    neg_sample_prob = label_counter[1] / label_counter[0]

    # for i in range(len(features)):
    #     print(f"Feature {i + 1}: {features[i]}")
    #     print(f"Label {i + 1}: {labels[i]}")

    # 假设你已经有了features和labels
    # 将features和labels转换为numpy数组
    features_np = np.concatenate(features_list)
    labels_np = np.concatenate(labels_list)

    # 将numpy数组转换为PyTorch张量
    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_np, dtype=torch.float32)

    # 创建数据集（Dataset）和数据加载器（DataLoader）
    batch_size = 128
    dataset = TensorDataset(features_tensor, labels_tensor)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=CustomSampler(dataset, neg_sample_prob),
                             num_workers=4, drop_last=True)

    # 定义神经网络、损失函数和优化器
    input_size = len(features_tensor[0])
    hidden_size = 128
    output_size = 1
    model = FiveLayerNN(input_size, hidden_size, output_size)
    model.to(device)
    # 这里可以载入旧模型，继续训练
    # model_weights_path = f"model/buy_weights_lr_20240124_last.pth"
    # model.load_state_dict(torch.load(model_weights_path))
    # 调整正样本权重
    # pos_weight = torch.tensor([1.0])
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

    # 训练循环
    epochs = 3000
    for epoch in range(epochs):
        for batch_features, batch_labels in data_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            # 前向传播
            output = model(batch_features).squeeze()
            # 计算损失
            loss = loss_function(output, batch_labels)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 更新权重
            optimizer.step()
        # # 更新学习率
        # scheduler.step()

        if epoch % 10 == 1:
            # 打印每个epoch的损失
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")
        if epoch % 500 == 0:
            # 训练模型后，保存权重
            model_weights_path = f"{model_name}_{epoch}.pth"
            torch.save(model.state_dict(), model_weights_path)

    # 训练模型后，保存权重
    model_weights_path = f"{model_name}_last.pth"
    print(model_weights_path)
    torch.save(model.state_dict(), model_weights_path)

    # 计算训练集上的准确率
    with torch.no_grad():
        output = model(features_tensor).squeeze()
        predictions = (torch.sigmoid(output) > 0.5).float()
        accuracy = (predictions == labels_tensor).float().mean().item()
        print(f"Training accuracy: {accuracy * 100:.2f}%")

    # 打印每条样本的预测值
    # for i, prediction in enumerate(predictions):
    #     print(f"Sample {i + 1}: {prediction.item()}")

    # 将预测值转换为二进制值（0或1）
    binary_predictions = (predictions > 0).float()

    # 统计正样本（1）和负样本（0）的分布
    prediction_counter = Counter(binary_predictions.numpy())

    print("Prediction distribution:")
    print(f"Positive predictions (1): {prediction_counter[1.0]}")
    print(f"Negative predictions (0): {prediction_counter[0.0]}")

    # 筛选出预测结果为 1 的数据
    positive_predictions = predictions[predictions == 1]
    positive_labels = labels_tensor[predictions == 1]

    # 计算仅在预测结果为 1 时的准确率
    if len(positive_predictions) > 0:
        positive_accuracy = (positive_predictions == positive_labels).float().mean().item()
        print(f"Accuracy for predictions = 1: {positive_accuracy * 100:.2f}%")
    else:
        print("No positive predictions.")

    # 筛选出label为 1 的数据
    positive_predictions = predictions[labels_tensor == 1]
    positive_labels = labels_tensor[labels_tensor == 1]

    positive_recall = (positive_predictions == positive_labels).float().mean().item()
    print(f"Recall for predictions = 1: {positive_recall * 100:.2f}%")
