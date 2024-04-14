import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from net import Cnn
from last_dataset_creation import create_dataset

from debug import debug
import sys
import os


def split_tensor(tensor: torch.Tensor, split_ratio: float = 0.8, dimension: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    dim_size = tensor.size(dimension)
    split_size = int(dim_size * split_ratio)

    tensor1 = tensor.narrow(dimension, 0, split_size)
    tensor2 = tensor.narrow(dimension, split_size, dim_size - split_size)

    return tensor1, tensor2


def train(dataloader: DataLoader, model: nn.Module, loss_fn: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer,
          batch_size: int, log_interval: int = 3200):

    size = len(dataloader.dataset)
    model.train()

    data: torch.Tensor

    for batch_idx, (data, target) in enumerate(dataloader):
        if data.size(0) != batch_size or target.size(0) != batch_size:
            continue
        pred = model(data.unsqueeze(1))
        target = target.long()
        loss = loss_fn(pred, target)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % log_interval == 0:
            loss, current = loss.item(), batch_idx * batch_size + len(data)
            print(f'loss: {loss:>7f} [{current::>5d}/{size:>5d}]')


def test(model: nn.Module, dataloader: DataLoader, loss_fn: nn.CrossEntropyLoss, batch_size: int) -> float:
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data, target in dataloader:
            if data.size(0) != batch_size or target.size(0) != batch_size:
                continue
            pred = model(data.unsqueeze(1))
            target = target.long()
            test_loss += loss_fn(pred, target).item()
            correct += (pred.argmax(1) == target).float().sum().item()

    test_loss /= num_batches
    correct /= size
    print(f'Test error: \n Accuracy: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f} \n')
    return 100*correct


def execute_model(learning_rate: float, batch_size: int, n_epochs: int, log_interval: int = 3200, complete: bool = True):

    dataset = create_dataset(load=True, complete=complete)

    train_path = 'datasets/train_dataset_complete.pickle'
    test_path = 'datasets/test_dataset_complete.pickle'

    if os.path.exists(train_path) and os.path.exists(test_path):
        with open(train_path, 'rb') as file:
            train_loader = pickle.load(file)
        with open(test_path, 'rb') as file:
            test_loader = pickle.load(file)
    else:
        complete_train, complete_test = random_split(dataset, [0.8, 0.2])
        train_loader = DataLoader(complete_train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(complete_test, batch_size=batch_size, shuffle=True)

        with open(train_path, 'wb') as file:
            pickle.dump(train_loader, file)
        with open(test_path, 'wb') as file:
            pickle.dump(test_loader, file)

    net = Cnn()
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()


    #TODO: Qui dovrebbe andarci training e test
    #? come si deve fare il train, devo fare due training diversi per clean e noisy?
    avg = 0
    for epoch in range(n_epochs):
        print(f'Epoch: {epoch+1}\n')
        train(dataloader=train_loader, model=net, loss_fn=criterion, optimizer=optimizer, batch_size=batch_size, log_interval=log_interval)
        avg += test(dataloader=test_loader, model=net, loss_fn=criterion, batch_size=batch_size)
    print(f'-- AVG ACCURACY: {avg/n_epochs}%')
