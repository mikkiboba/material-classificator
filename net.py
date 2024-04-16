import torch.nn as nn
import torch.nn.functional as F
import torch
from debug import debug


class Cnn(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=256*8, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=13)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x

