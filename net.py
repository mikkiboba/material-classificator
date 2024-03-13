import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool = nn.MaxPool1d(
            kernel_size=2,
            stride=2
        )
        
        with torch.no_grad():
            self.dummy_input = torch.zeros((1,14432,128))
            output_size = self.forward_features_size_calculation(self.dummy_input)
        
        self.fc1 = nn.Linear(
            in_features=output_size,
            out_features=512
        )
        self.fc2 = nn.Linear(
            in_features=512,
            out_features=13
        )
    
    def forward(self,x):
        x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def forward_features_size_calculation(self,x):
        x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x.nelement() // x.shape[0]
        
    