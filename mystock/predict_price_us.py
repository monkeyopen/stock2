from dotenv import load_dotenv
import os

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
CONF_PATH = os.getenv('CONF_PATH')
ROOT_PATH = os.getenv('ROOT_PATH')
MODEL_PATH = os.getenv('MODEL_PATH')
import sys

# print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([ROOT_PATH])
import datetime
import numpy as np
import torch
from get_fortune import Backtesting
from mynet.neural_network import FiveLayerNN
import read


def predict_price(dataset="us_stock_follow",
                  model_weights_path="model/mse_loss_us_stock_price_step50_1_20180101_20231231_last.pth",
                  flag=False, label_type="price", start_date="20240201",
                  end_date="20241231"):
    label_file = CONF_PATH + dataset
    print(model_weights_path)
    features_list = []
    labels_list = []
    infos_list = []
    pre_list = []
    result = []
    all_stock = read.read_us_name()
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
                end_date=datetime.datetime(2024, 12, 31),
                sample_start=datetime.datetime.strptime(start_date, "%Y%m%d"),
                sample_end=datetime.datetime.strptime(end_date, "%Y%m%d")
            )
            backtest._read_data()
            feature, label, info, pre = backtest.generate_samples(label_type=label_type)
            if len(feature) < 1:
                continue

            # for i in range(len(feature)):
            #     print(feature[i])
            # print(feature[0][100:])
            # print(feature[1][100:])
            # myfea = np.zeros_like(feature[0])
            # myfea[:] = feature[0][:]
            # myfea[120:121] = feature[0][120:121]
            # print(myfea[100:])

            # 将features和labels转换为numpy数组
            # features_np = np.array([myfea])
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
                predictions = model(features_tensor).squeeze()
            # 遍历并打印每一条数据的预测结果和标签
            for i in range(len(predictions)):
                print(
                    f"{stock},{all_stock.get(stock)}, Pre {predictions[i].item():.4f}, Label {labels_tensor[i].item():.4f}, info {info[i]}")

            result.append(
                f"Pre {predictions[-1].item():.4f}, Label {labels_tensor[-1].item():.4f}, info {info[-1]}, {stock},{all_stock.get(stock)}")

        return result


if __name__ == '__main__':
    model_weights_path_list = [
        # "model/mse_side_loss_us_stock_price_step50_1_20180101_20231231_last.pth"
        # "model/smooth_l1_loss_us_stock_price_step50_1_20180101_20231231_last.pth"
        f"{MODEL_PATH}/l1_loss_a_stock_price_step50_1_20180101_20231231_last.pth"
    ]
    result_list = []
    for model_weights_path in model_weights_path_list:
        result = predict_price(dataset="us_stock_follow", model_weights_path=model_weights_path,
                               flag=False, label_type="price", start_date="20240201",
                               end_date="20241231")
        # result_list.append((name.split("_")[0], pre, label, info))
        print()
        for i in result:
            print(i)
        # result_list.append((name, pre, label, info))

        # # 获取最长的 pre 和 label 列表长度
        # max_pre_len = max(len(pre) for _, pre, _, _ in result_list)
        #
        # # 打印结果
        # for i in range(max_pre_len):
        #     print(f"{info[i]}, ", end="\t")
        #     for name, pre, label, infos in result_list:
        #         if i < len(pre):
        #             print(f"{name}: {pre[i]:.3f}, {label[i]:.3f}, {infos[i][0]}", end="\t")
        #         else:
        #             print("-" * 20, end="\t")
        #     print()
