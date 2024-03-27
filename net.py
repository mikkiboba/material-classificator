import torch.nn as nn
import torch.nn.functional as F
import torch
from debug import debug
import math


class NNet(nn.Module):
    def __init__(self, in_channels_train: int, in_channels_test: int, kernel_size: int = 3):
        super(NNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc_train = nn.Linear(in_features=(math.floor(in_channels_train/4) - (kernel_size-1)) * 32, out_features=13) # 2886 - (3-1) = 2886 - 2 = 2884 * 32 = 92288
        self.fc_test = nn.Linear(in_features=(math.ceil(in_channels_test/4) - (kernel_size-1)) * 32, out_features=13)
        self.dropout = nn.Dropout1d()

    def forward(self, x: torch.Tensor):
        x = torch.permute(x, (0, 2, 1))

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)

        try:
            x = self.fc_train(x)
        except RuntimeError:
            x = self.fc_test(x)
        x = self.dropout(x)

        x = F.softmax(x, dim=0)

        return x

