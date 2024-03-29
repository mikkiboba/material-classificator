import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from net import NNet
from new_dataset_creation import create_dataset

from debug import debug


def split_tensor(tensor: torch.Tensor, split_ratio: float = 0.8, dimension: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    dim_size = tensor.size(dimension)
    split_size = int(dim_size * split_ratio)

    dataset1 = tensor.narrow(dimension, 0, split_size)
    dataset2 = tensor.narrow(dimension, split_size, dim_size - split_size)

    return dataset1, dataset2


def train(net: nn.Module, train_loader: DataLoader, optimizer: optim.Adam, log_interval: int, epoch: int,
          train_losses: list, train_counter: list, criterion: nn.CrossEntropyLoss) -> None:
    net.train()


def test(net: nn.Module, test_loader: DataLoader, criterion: nn.Module) -> None:
    net.eval()


def main() -> None:
    # Parameters
    learning_rate = 0.01
    batch_size = 1
    n_epochs = 10
    log_interval = 1
    path = 'data'
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl = create_dataset(plot=False, load=True)

    clean_tensor = torch.Tensor()
    noisy_tensor = torch.Tensor()
    labels_tensor = torch.Tensor()

    for clean, noisy, labels in dl:
        clean_tensor = clean.clone().detach()
        noisy_tensor = noisy.clone().detach()
        labels_tensor = labels.clone().detach()

    clean_train, clean_test = split_tensor(clean_tensor)
    noisy_train, noisy_test = split_tensor(noisy_tensor)

    net = NNet(clean_train.size(1), clean_test.size(1))

    train_dataset = TensorDataset(clean_train, noisy_train, labels_tensor)
    test_dataset = TensorDataset(clean_test, noisy_test, labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_counter = []

    test(net, test_loader, criterion)
    for epoch in range(1, n_epochs+1):
        train(net, train_loader, optimizer, log_interval, epoch, train_losses, train_counter, criterion)
        test(net, test_loader, criterion)


if __name__ == "__main__":
    now = time.time()
    main()
    print(f'-- EXECUTION TIME: {time.time()-now} seconds')
