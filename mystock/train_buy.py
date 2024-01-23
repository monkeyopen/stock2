import pandas as pd

from dotenv import load_dotenv
import os
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from get_fortune import Backtesting
from mynet.neural_network import FiveLayerNN
from collections import Counter

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')

if __name__ == '__main__':
    features_list = []
    labels_list = []
    infos_list = []
    label_file = DATA_PATH + "us_stock"
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
                sample_start=datetime.datetime(2010, 1, 1),
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
    batch_size = 100
    dataset = TensorDataset(features_tensor, labels_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 定义神经网络、损失函数和优化器
    input_size = len(features_tensor[0])
    hidden_size = 128
    output_size = 1
    model = FiveLayerNN(input_size, hidden_size, output_size)
    # 这里可以载入旧模型，继续训练
    # model_weights_path = f"model/20240121_buy_weights.pth"
    # model.load_state_dict(torch.load(model_weights_path))

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    epochs = 3000
    for epoch in range(epochs):
        for batch_features, batch_labels in data_loader:
            # 前向传播
            output = model(batch_features).squeeze()

            # 计算损失
            loss = loss_function(output, batch_labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 更新权重
            optimizer.step()

        if epoch % 10 == 1:
            # 打印每个epoch的损失
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")
        if epoch % 500 == 0:
            # 训练模型后，保存权重
            model_weights_path = f"model/buy_weights_20240123_{epoch}.pth"
            torch.save(model.state_dict(), model_weights_path)

    # 训练模型后，保存权重
    model_weights_path = f"model/20240121_buy_weights.pth"
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
    binary_predictions = (predictions > 0.5).float()

    # 统计正样本（1）和负样本（0）的分布
    prediction_counter = Counter(binary_predictions.numpy())

    print("Prediction distribution:")
    print(f"Positive predictions (1): {prediction_counter[1.0]}")
    print(f"Negative predictions (0): {prediction_counter[0.0]}")
