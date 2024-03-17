import torch.nn as nn
import torch.nn.functional as F
import torch
from debug import debug


class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=5770, out_features=13) #out = numero di materiali

    def forward(self, x: torch.Tensor):
        debug(x.shape)
        x = torch.permute(x, (0, 2, 1))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
