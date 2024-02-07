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


class StockScore:
    def __init__(self, name, acc, recall):
        self.name = name
        self.acc = float(acc)
        self.recall = float(recall)

    def __str__(self):
        return f"{self.name},{self.acc}, {self.recall}"


def select_stock(dataset="us_stock_test2", model_weights_path="model/sell_240_step300_20130101_20231231_last.pth",
                 flag=False, label_type="buy1", start_date="20240101",
                 end_date="20241231"):
    label_file = CONF_PATH + dataset
    # model_weights_path = f"model/sell_weights_test_20180101_20231231_last.pth"
    print(model_weights_path)
    features_list = []
    labels_list = []
    infos_list = []
    pre_list = []
    stock_name = []
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
                sample_start=datetime.datetime.strptime(start_date, "%Y%m%d"),
                sample_end=datetime.datetime.strptime(end_date, "%Y%m%d")
            )
            backtest._read_data()
            feature, label, info, pre = backtest.generate_samples(label_type=label_type)
            if len(feature) < 1:
                continue

            label_counter = Counter(label)
            print("Label distribution:")
            print(f"Positive samples (1): {label_counter[1]}")
            print(f"Negative samples (0): {label_counter[0]}")

            # for i in range(len(features)):
            #     print(f"Feature {i + 1}: {features[i]}")
            #     print(f"Label {i + 1}: {labels[i]}")

            # 假设你已经有了features和labels
            # 将features和labels转换为numpy数组
            features_np = np.array(feature)
            labels_np = np.array(label)
            pres_np = np.array(pre)

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
            #         f"Data {i + 1}: output {output[i].item():.2f}, Pre {predictions[i].item()}, Label {labels_tensor[i].item()}, info {info[i]}")

            # 将预测值转换为二进制值（0或1）
            binary_predictions = (predictions > 0).float()

            # 统计正样本（1）和负样本（0）的分布
            prediction_counter = Counter(binary_predictions.numpy())

            print("Prediction distribution:")
            print(f"Positive predictions (1): {prediction_counter[1.0]}")
            print(f"Negative predictions (0): {prediction_counter[0.0]}")

            calAcc("规则", pres_tensor, labels_tensor)
            calRecall("规则", pres_tensor, labels_tensor)
            acc = calAcc("模型", predictions, labels_tensor)
            recall = calRecall("模型", predictions, labels_tensor)
            stock_name.append(StockScore(stock, acc, recall))

        filtered_stock_name = [stock for stock in stock_name if stock.recall > 0.5]
        sorted_allStock = sorted(filtered_stock_name, key=lambda StockScore: (StockScore.acc, StockScore.recall),
                                 reverse=True)
        for stock in sorted_allStock[:10]:
            print(stock)


if __name__ == '__main__':
    select_stock(dataset="us_stock", model_weights_path="model/us_stock_sell5_step100_05_20180101_20231231_200.pth",
                 flag=False, label_type="sell5", start_date="20240101",
                 end_date="20241231")
    # buy1
    # us_stock_buy1_step100_05_20180101_20231231_100.pth
    # buy5
    # us_stock_buy5_step100_05_20180101_20231231_last.pth
    # sell1
    # us_stock_sell1_step100_05_20180101_20231231_100.pth
    # sell5
    # us_stock_sell5_step100_05_20180101_20231231_200.pth
