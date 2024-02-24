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


def test_price_stock(stock="TSLA", model_weights_path="model/sell_240_step100_20130101_20231231_last.pth",
                     flag=False, label_type="buy1", start_date="20240101",
                     end_date="20241231"):
    # model_weights_path = f"model/sell_weights_test_20180101_20231231_last.pth"
    print(model_weights_path)
    features_list = []
    labels_list = []
    infos_list = []
    pre_list = []
    stock_name = []
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
        return 0, 0, 0, 0
    if len(feature) > 0:
        features_list.append(feature)
        labels_list.append(label)
        infos_list.append(info)
        pre_list.append(pre)

    features_np = np.concatenate(features_list)
    labels_np = np.concatenate(labels_list)
    pres_np = np.concatenate(pre_list)
    info_np = np.concatenate(infos_list)

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
    output_size = 1
    random_signal = True
    # 创建一个新的模型实例
    model = FiveLayerNN(input_size, hidden_size, output_size)
    if model_weights_path != "random":
        random_signal = False
        # 加载模型状态
        model.load_state_dict(torch.load(model_weights_path))
    # print(model.state_dict())
    # 确保模型在评估模式，这会关闭诸如 dropout 等训练特定的层
    model.eval()
    buy_signal = 1
    sell_signal = 1
    stop_loss = 0.98
    mse, mae, acc, predictions = testPrice(model, features_tensor, labels_tensor, buy_signal, sell_signal,
                                           random_signal)
    # 买卖的信号的阈值限定，止损策略的阈值设定
    backtest._set_signals(predictions, buy_signal, sell_signal)
    backtest._calculate_profit(stop_loss)

    profit = backtest.profit - 1
    print(model_weights_path)
    print(stock)
    print(f"{profit * 100:.2f}%")
    return mse, mae, acc, profit


if __name__ == '__main__':
    model_weights_path_list = [
        # "model/mse_side_loss_us_stock_price_step50_1_20180101_20231231_last.pth",
        "model/mse_side_loss_a_stock_price_step50_1_20180101_20231231_last.pth"]
    result_list = []
    label_file = CONF_PATH + "us_stock_test"
    for model_weights_path in model_weights_path_list:
        with open(label_file, 'r') as f:
            for line in f.readlines():
                stock = line.strip()
                # print(stock)
                if len(stock) < 1:
                    continue
                mse, mae, acc, profit = test_price_stock(stock=stock, model_weights_path=model_weights_path,
                                                         flag=False, label_type="price", start_date="20240201",
                                                         end_date="20241231")
                result_list.append((model_weights_path[5:20], stock, mse, mae, acc, profit))

    sorted_result = sorted(result_list, key=lambda x: float(x[5]), reverse=True)

    # 使用字典来聚合每个key的前10个元素
    aggregated_data = {}
    for item in sorted_result:
        key = item[0]
        if key not in aggregated_data:
            aggregated_data[key] = []
        aggregated_data[key].append(item)

    # 取出每个key的前10个元素
    model_result = []
    for key, items in aggregated_data.items():
        top_items = items[:]
        #     计算每个key的平均值
        mse_avg = sum([float(item[2]) for item in top_items]) / len(top_items)
        mae_avg = sum([float(item[3]) for item in top_items]) / len(top_items)
        acc_avg = sum([float(item[4]) for item in top_items]) / len(top_items)
        profit_avg = sum([float(item[5]) for item in top_items]) / len(top_items)
        model_result.append((key, mse_avg, mae_avg, acc_avg, profit_avg))
        # print(f"{key}: mse={mse_avg:.4f}, mae={mae_avg:.4f}, acc={acc_avg:.4f}, profit={profit_avg:.4f}")

    sorted_model = sorted(model_result, key=lambda x: float(x[4]), reverse=True)
    for result in sorted_model:
        print(f"{result[0]}: mse={result[1]:.4f}, mae={result[2]:.4f}, acc={result[3]:.4f}, profit={result[4]:.4f}")
    print("=============================")
    for result in sorted_result[:20]:
        # print(result)
        print(
            f"{result[0]},{result[1]}, acc={result[4]:.4f}, profit={result[5]:.4f}")
