import torch
from collections import Counter


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


def calRecall(name, predictions, labels_tensor):
    # 筛选出label为 1 的数据
    positive_predictions = predictions[labels_tensor == 1]
    positive_labels = labels_tensor[labels_tensor == 1]

    positive_recall = (positive_predictions == positive_labels).float().mean().item()
    print(f"{name} Recall for predictions = 1: {positive_recall * 100:.2f}%")


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
