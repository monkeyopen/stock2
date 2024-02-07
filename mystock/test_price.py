from dotenv import load_dotenv
import os
import datetime
import numpy as np
import torch
from get_fortune import Backtesting
from mynet.neural_network import FiveLayerNN
from collections import Counter

from mystock.op.metric import calAcc, calRecall, testPrice

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
CONF_PATH = os.getenv('CONF_PATH')


def test_price(dataset="us_stock_test2", model_weights_path="model/sell_240_step100_20130101_20231231_last.pth",
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
            # print(stock)
            if len(stock) < 1:
                continue
            # if stock != "PYPL":
            #     continue
            stock_path = DATA_PATH + stock
            backtest = Backtesting(
                data_dir=stock_path,
                dt_format='%Y-%m-%d',
                start_date=datetime.datetime(2023, 7, 1),
                end_date=datetime.datetime(2024, 12, 31),
                sample_start=datetime.datetime.strptime(start_date, "%Y%m%d"),
                sample_end=datetime.datetime.strptime(end_date, "%Y%m%d")
            )
            backtest._read_data()
            feature, label, info, pre = backtest.generate_samples(label_type=label_type)
            if len(feature) < 1:
                continue
            if len(feature) > 0:
                features_list.append(feature)
                labels_list.append(label)
                infos_list.append(info)
                pre_list.append(pre)

        features_np = np.concatenate(features_list)
        labels_np = np.concatenate(labels_list)
        pres_np = np.concatenate(pre_list)
        info_np = np.concatenate(infos_list)

        # for i in range(len(features)):
        #     print(f"Feature {i + 1}: {features[i]}")
        #     print(f"Label {i + 1}: {labels[i]}")

        # 假设你已经有了features和labels
        # 将features和labels转换为numpy数组
        # features_np = np.array(features)
        # labels_np = np.array(labels)
        # pres_np = np.array(pres)

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
        # 确保模型在评估模式，这会关闭诸如 dropout 等训练特定的层
        model.eval()
        mse, mae, acc, predictions = testPrice(model, features_tensor, labels_tensor)
        backtest._set_signals(predictions, buy_signal=1, sell_signal=1)
        backtest._calculate_profit(stop_loss=1)

        profit = backtest.profit - 1
        print(model_weights_path)
        print(dataset)
        print(f"{profit * 100:.2f}%")
        return mse, mae, acc, profit


if __name__ == '__main__':
    model_weights_path_list = [
        "model/log_cosh_loss_us_stock_price_step100_05_20180101_20231231_last.pth",
        "model/huber_loss_us_stock_price_step100_05_20180101_20231231_last.pth",
        "model/l1_loss_us_stock_price_step100_05_20180101_20231231_last.pth",
        "model/smooth_l1_loss_us_stock_price_step100_05_20180101_20231231_last.pth",
        "model/mse_loss_us_stock_price_step100_05_20180101_20231231_100.pth"]
    result_list = []
    for model_weights_path in model_weights_path_list:
        mse, mae, acc, profit = test_price(dataset="us_stock_INTC", model_weights_path=model_weights_path,
                                           flag=False, label_type="price", start_date="20240101",
                                           end_date="20241231")
        result_list.append(
            (model_weights_path, f"{mse * 100:.2f}%", f"{mae * 100:.2f}%", f"{acc * 100:.2f}%", f"{profit * 100:.2f}%"))

    for result in result_list:
        print(result)
