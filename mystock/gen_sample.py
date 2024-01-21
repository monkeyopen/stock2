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


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def generate_samples(df, window_size):
    features = []
    labels = []
    close_prices = df['close'].values
    volume = df['volume'].values
    feature_size = 0

    # 循环遍历生成样本数据，一次循环生成一条数据，所以循环的次数是样本窗口大小。单词循环中会读取特征需要的数据大小。
    for i in range(window_size + 29, len(df) - 1):
        # 提取close价格作为特征
        feature_close = close_prices[i - window_size - 29:i]
        feature_close_normalized = feature_close / feature_close[-1]

        # 提取成交量作为特征
        feature_volume = volume[i - window_size - 29:i]
        feature_volume_normalized = feature_volume / feature_volume[-1]

        # 计算MA5
        feature_ma5 = moving_average(feature_close_normalized, 5)
        # 计算MA10
        feature_ma10 = moving_average(feature_close_normalized, 10)
        # 计算MA20
        feature_ma20 = moving_average(feature_close_normalized, 20)
        # 计算MA30
        feature_ma30 = moving_average(feature_close_normalized, 30)

        # 将所有特征添加到一个列表中
        feature_list = [feature_close_normalized[-window_size:], feature_volume_normalized[-window_size:],
                        feature_ma5[-window_size:], feature_ma10[-window_size:],
                        feature_ma20[-window_size:], feature_ma30[-window_size:]]

        # 使用 np.concatenate() 函数将特征列表连接起来
        feature = np.concatenate(feature_list)

        feature_size = feature.size
        # print(df['close'].iloc[i + 1], df['open'].iloc[i + 1])
        label = int(df['close'].iloc[i + 1] > df['open'].iloc[i + 1] * 1.01)
        # label = int(df['close'].iloc[i - 1] > df['close'].iloc[i - 2])
        # label = int(normalized_close_feature[-1] > normalized_close_feature[-2])

        features.append(feature)
        labels.append(label)

    return features, labels, feature_size


if __name__ == '__main__':
    stock_path = DATA_PATH + "/00700"
    backtest = Backtesting(
        data_dir=stock_path,
        dt_format='%Y-%m-%d',
        start_date=datetime.datetime(2023, 1, 1),
        end_date=datetime.datetime(2023, 12, 31)
    )
    backtest._read_data()

    window_size = 20

    features, labels, feature_size = generate_samples(backtest.df, window_size)

    label_counter = Counter(labels)
    print("Label distribution:")
    print(f"Positive samples (1): {label_counter[1]}")
    print(f"Negative samples (0): {label_counter[0]}")

    # for i in range(len(features)):
    #     print(f"Feature {i + 1}: {features[i]}")
    #     print(f"Label {i + 1}: {labels[i]}")

    # 假设你已经有了features和labels
    # 将features和labels转换为numpy数组
    features_np = np.array(features)
    labels_np = np.array(labels)

    # 将numpy数组转换为PyTorch张量
    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_np, dtype=torch.float32)

    # 创建数据集（Dataset）和数据加载器（DataLoader）
    batch_size = 50
    dataset = TensorDataset(features_tensor, labels_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 定义神经网络、损失函数和优化器
    input_size = feature_size
    hidden_size = 128
    output_size = 1
    model = FiveLayerNN(input_size, hidden_size, output_size)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    epochs = 1000
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
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

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
