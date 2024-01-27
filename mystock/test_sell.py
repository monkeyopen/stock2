from dotenv import load_dotenv
import os
import datetime
import numpy as np
import torch
from get_fortune import Backtesting
from mynet.neural_network import FiveLayerNN
from collections import Counter

from mystock.op.metric import calAcc, calRecall

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
CONF_PATH = os.getenv('CONF_PATH')

if __name__ == '__main__':
    label_file = CONF_PATH + "us_stock_test2"
    model_weights_path = f"model/sell_weights_20230101_20231231_last.pth"
    print(model_weights_path)
    with open(label_file, 'r') as f:
        for line in f.readlines():
            stock = line.strip()
            print(stock)
            if len(stock) < 1:
                continue
            # if stock != "PYPL":
            #     continue
            stock_path = DATA_PATH + stock
            backtest = Backtesting(
                data_dir=stock_path,
                dt_format='%Y-%m-%d',
                start_date=datetime.datetime(2023, 1, 1),
                end_date=datetime.datetime(2024, 1, 31),
                sample_start=datetime.datetime(2024, 1, 1),
                sample_end=datetime.datetime(2024, 1, 31)
            )
            backtest._read_data()
            features, labels, infos, pres = backtest.generate_samples(label_type="sell5")
            if len(features) < 1:
                continue

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
            pres_np = np.array(pres)

            # 将numpy数组转换为PyTorch张量
            features_tensor = torch.tensor(features_np, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_np, dtype=torch.float32)
            pres_tensor = torch.tensor(pres_np, dtype=torch.float32)

            # 定义神经网络、损失函数和优化器
            input_size = len(features_tensor[0])
            hidden_size = 128
            output_size = 1
            # 创建一个新的模型实例
            model = FiveLayerNN(input_size, hidden_size, output_size)
            # 加载模型状态
            model.load_state_dict(torch.load(model_weights_path))
            # print(model.state_dict())
            # model.load_state_dict(torch.load("model/sell_weights_20130101_20231231_500.pth"))
            # print(model.state_dict())
            # model.load_state_dict(torch.load("model/sell_weights_20130101_20231231_1000.pth"))
            # print(model.state_dict())
            # 确保模型在评估模式，这会关闭诸如 dropout 等训练特定的层
            model.eval()

            # 计算训练集上的准确率
            with torch.no_grad():
                output = model(features_tensor).squeeze()
                output = torch.sigmoid(output)
                predictions = (output > 0.5).float()
                accuracy = (predictions == labels_tensor).float().mean().item()
                print(f"Test accuracy: {accuracy * 100:.2f}%")

            # # 遍历并打印每一条数据的预测结果和标签
            # for i in range(len(predictions)):
            #     print(
            #         f"Data {i + 1}: output {output[i].item():.2f}, Pre {predictions[i].item()}, Label {labels_tensor[i].item()}, info {infos[i]}")

            # 将预测值转换为二进制值（0或1）
            binary_predictions = (predictions > 0).float()

            # 统计正样本（1）和负样本（0）的分布
            prediction_counter = Counter(binary_predictions.numpy())

            print("Prediction distribution:")
            print(f"Positive predictions (1): {prediction_counter[1.0]}")
            print(f"Negative predictions (0): {prediction_counter[0.0]}")

            calAcc("规则", pres_tensor, labels_tensor)
            calRecall("规则", pres_tensor, labels_tensor)
            calAcc("模型", predictions, labels_tensor)
            calRecall("模型", predictions, labels_tensor)
