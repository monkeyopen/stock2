import torch
import torch.nn as nn
import torch.nn.functional as F


class FiveLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FiveLayerNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.layer3 = nn.Linear(int(hidden_size / 2), int(hidden_size / 4))
        self.layer4 = nn.Linear(int(hidden_size / 4), int(hidden_size / 8))
        self.layer5 = nn.Linear(int(hidden_size / 8), output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.layer5(x)
        return x


if __name__ == '__main__':
    # 示例
    input_size = 10
    hidden_size = 20
    output_size = 5

    model = FiveLayerNN(input_size, hidden_size, output_size)
    input_data = torch.randn(1, input_size)
    output_data = model(input_data)

    print("Input data:", input_data)
    print("Output data:", output_data)
