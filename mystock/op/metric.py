import torch
from collections import Counter
import torch.nn.functional as F


def calAcc(name, predictions, labels_tensor):
    # 筛选出预测结果为 1 的数据
    positive_predictions = predictions[predictions == 1]
    positive_labels = labels_tensor[predictions == 1]

    # 计算仅在预测结果为 1 时的准确率
    if len(positive_predictions) > 0:
        positive_accuracy = (positive_predictions == positive_labels).float().mean().item()
        print(f"{name} Accuracy for predictions = 1: {positive_accuracy * 100:.2f}%")
    else:
        print(f"{name} No positive predictions.")
        positive_accuracy = 0

    return positive_accuracy


def calRecall(name, predictions, labels_tensor):
    # 筛选出label为 1 的数据
    positive_predictions = predictions[labels_tensor == 1]
    positive_labels = labels_tensor[labels_tensor == 1]

    positive_recall = (positive_predictions == positive_labels).float().mean().item()
    print(f"{name} Recall for predictions = 1: {positive_recall * 100:.2f}%")

    return positive_recall


def testSample(model, features_tensor, labels_tensor, pres_tensor):
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

    calAcc("规则", pres_tensor, labels_tensor)
    calRecall("规则", pres_tensor, labels_tensor)
    calAcc("模型", predictions, labels_tensor)
    calRecall("模型", predictions, labels_tensor)


def testPrice(model, features_tensor, labels_tensor, buy_signal=1, sell_signal=1):
    # 计算训练集上的准确率
    with torch.no_grad():
        predictions = model(features_tensor).squeeze()
        # 计算均方误差（MSE）
        mse_loss = F.mse_loss(predictions, labels_tensor)
        print(f"Mean Squared Error: {mse_loss.item()}")

        # 计算平均绝对误差（MAE）
        mae_loss = F.l1_loss(predictions, labels_tensor)
        print(f"Mean Absolute Error: {mae_loss.item()}")

        # 计算 R² 分数
        residuals = labels_tensor - predictions
        ss_total = torch.sum((labels_tensor - labels_tensor.mean()) ** 2)
        ss_residual = torch.sum(residuals ** 2)
        r2_score = 1 - (ss_residual / ss_total)
        print(f"R² Score: {r2_score.item()}")

        # 计算预测值和标签都大于1或都小于1的一致性
        consistency_count = 0
        total_count = 0

        for pred, label in zip(predictions, labels_tensor):
            if (pred > buy_signal and label > 1) or (pred < sell_signal and label < 1):
                consistency_count += 1
            total_count += 1

        consistency_ratio = consistency_count / total_count
        print(f"Consistency ratio: {consistency_ratio * 100:.2f}%")

    return mse_loss.item(), mae_loss.item(), consistency_ratio, predictions
