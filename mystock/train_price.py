from dotenv import load_dotenv
import os

load_dotenv()
CONF_PATH = os.getenv('CONF_PATH')
DATA_PATH = os.getenv('DATA_PATH')
ROOT_PATH = os.getenv('ROOT_PATH')
MODEL_PATH = os.getenv('MODEL_PATH')
import sys

# print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([ROOT_PATH])
from mystock.op.metric import testPrice
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from get_fortune import Backtesting
from mynet.neural_network import FiveLayerNN
from mystock.op.loss import mse_side_loss
from collections import Counter

if __name__ == '__main__':
    features_list = []
    labels_list = []
    infos_list = []
    pre_list = []
    datafile = "hk_stock"
    label_file = CONF_PATH + datafile
    label_type = "price"
    start_date = "20180101"
    end_date = "20231231"
    # smooth_l1_loss，mse_loss，l1_loss，huber_loss，log_cosh_loss, mse_side_loss
    loss = "mse_side_loss"
    step_size = 50
    gamma = 0.1
    model_name = f"{MODEL_PATH}/{loss}_{datafile}_{label_type}_step{step_size}_{gamma * 10:.0f}_{start_date}_{end_date}"
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')
    with open(label_file, 'r') as f:
        for line in f.readlines():
            stock = line.strip()
            if len(stock) < 1:
                continue
            print(stock)
            stock_path = DATA_PATH + stock
            backtest = Backtesting(
                data_dir=stock_path,
                dt_format='%Y-%m-%d',
                start_date=datetime.datetime(2017, 1, 1),
                end_date=datetime.datetime(2024, 12, 31),
                sample_start=datetime.datetime.strptime(start_date, "%Y%m%d"),
                sample_end=datetime.datetime.strptime(end_date, "%Y%m%d")
            )
            backtest._read_data()
            feature, label, info, pre = backtest.generate_samples(label_type=label_type)
            if len(feature) > 0:
                features_list.append(feature)
                labels_list.append(label)
                infos_list.append(info)
                pre_list.append(pre)
    # labels = np.concatenate(labels_list)
    # label_counter = Counter(labels)
    # print("Label distribution:")
    # print(f"Positive samples (1): {label_counter[1]}")
    # print(f"Negative samples (0): {label_counter[0]}")
    # neg_sample_prob = label_counter[1] / label_counter[0]

    # for i in range(len(features)):
    #     print(f"Feature {i + 1}: {features[i]}")
    #     print(f"Label {i + 1}: {labels[i]}")

    # 假设你已经有了features和labels
    # 将features和labels转换为numpy数组
    features_np = np.concatenate(features_list)
    labels_np = np.concatenate(labels_list)
    pres_np = np.concatenate(pre_list)

    # 将numpy数组转换为PyTorch张量
    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_np, dtype=torch.float32)
    pres_tensor = torch.tensor(pres_np, dtype=torch.float32)

    # 创建数据集（Dataset）和数据加载器（DataLoader）
    batch_size = 1024
    dataset = TensorDataset(features_tensor, labels_tensor)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # 定义神经网络、损失函数和优化器
    input_size = len(features_tensor[0])
    hidden_size = 128
    output_size = 1
    model = FiveLayerNN(input_size, hidden_size, output_size)


    # model.to(device)
    # 这里可以载入旧模型，继续训练
    # model_weights_path = f"{model_name}_1000.pth"
    # model.load_state_dict(torch.load(model_weights_path))
    # 调整正样本权重
    # pos_weight = torch.tensor([1.0])
    # loss_function = nn.BCEWithLogitsLoss()

    # 实现Log-Cosh损失
    def log_cosh_loss(input, target):
        loss = torch.log(torch.cosh(input - target))
        return loss.mean()


    if loss == "mse_loss":
        loss_function = nn.functional.mse_loss
    elif loss == "l1_loss":
        loss_function = nn.functional.l1_loss
    elif loss == "smooth_l1_loss":
        loss_function = nn.functional.smooth_l1_loss
    elif loss == "huber_loss":
        loss_function = nn.functional.huber_loss
    elif loss == "log_cosh_loss":
        loss_function = log_cosh_loss
    elif loss == "mse_side_loss":
        loss_function = mse_side_loss
    else:
        loss_function = nn.functional.mse_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 训练循环
    epochs = 200
    num = 0
    for epoch in range(epochs):
        for batch_features, batch_labels in data_loader:
            num += 1
            # batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            # 前向传播
            output = model(batch_features).squeeze()
            if torch.isnan(output).any():
                print(num, "output contains NaN values:", output)

            # # 遍历并打印output和batch_labels
            # for i, (out, feature, label) in enumerate(zip(output, batch_features, batch_labels)):
            #     print(f"num {num}, Sample {i}: output = {out}, label = {label}, feature = {feature}")
            # 计算损失
            loss = loss_function(output, batch_labels)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 更新权重
            optimizer.step()
        # 更新学习率
        scheduler.step()

        # 获取当前时间
        current_time = datetime.datetime.now()
        # 打印每个epoch的损失
        print(f"{current_time}, Epoch {epoch}/{epochs}, Loss: {loss.item()}")
        if epoch % 50 == 0:
            # 训练模型后，保存权重
            model_weights_path = f"{model_name}_{epoch}.pth"
            print(model_weights_path)
            torch.save(model.state_dict(), model_weights_path)
            testPrice(model, features_tensor, labels_tensor)

    # 训练模型后，保存权重
    model_weights_path = f"{model_name}_last.pth"
    print(model_weights_path)
    torch.save(model.state_dict(), model_weights_path)
    testPrice(model, features_tensor, labels_tensor)

    # test_sell(dataset=datafile, model_weights_path=model_weights_path,
    # flag=False, label_type=label_type)
