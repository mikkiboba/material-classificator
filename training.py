
#TODO: stesso training ma solo su clean e noisy
#TODO: controllare i giusti valori da mettere nei conv layers
#TODO: decidere ufficialmente come e quali valori cambiare per i vari test

import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset

from net import Cnn
from last_dataset_creation import create_dataset

from debug import debug
import sys
import os

def log(learning_rate: float, batch_size: int, epochs: int, accuracy: list[float], loss: list[float], train_dataset: str = "", test_dataset: str = ""):
    log_path = 'results.txt'
    avg_accuracy = sum(accuracy)/epochs
    avg_loss = sum(loss)/epochs
    string = f'{train_dataset=}\t{test_dataset=}\t{learning_rate=}\t{batch_size=}\t{epochs=}\t{accuracy=}\t{avg_accuracy=}\t{loss=}\t{avg_loss=}\n'
    with open(log_path, 'a') as file:
        file.write(string)


def split_tensor(tensor: torch.Tensor, split_ratio: float = 0.8, dimension: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    dim_size = tensor.size(dimension)
    split_size = int(dim_size * split_ratio)

    tensor1 = tensor.narrow(dimension, 0, split_size)
    tensor2 = tensor.narrow(dimension, split_size, dim_size - split_size)

    return tensor1, tensor2


def get_loaders(train_path: str, test_path:str) -> tuple[DataLoader, DataLoader]:
    with open(train_path, 'rb') as file:
        train_loader = pickle.load(file)
    with open(test_path, 'rb') as file:
        test_loader = pickle.load(file)
    return train_loader, test_loader
    
    
def make_loaders(dataset: Dataset, batch_size: int, train_path: str, test_path: str) -> tuple[DataLoader, DataLoader]:
    train, test = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
    
    with open(train_path, 'wb') as file:
        pickle.dump(train_loader, file)
    with open(test_path, 'wb') as file:
        pickle.dump(test_loader, file)
    
    return train_loader, test_loader


def execute_model(train_loader: DataLoader, test_loader: DataLoader, learning_rate: float, batch_size: int, n_epochs: int, log_interval: int) -> tuple[list[float], list[float]]:
    net = Cnn()
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    accuracy, loss = [], []
    for epoch in range(n_epochs):
        print(f'Epoch: {epoch+1}\n')
        train(dataloader=train_loader, model=net, loss_fn=criterion, optimizer=optimizer, batch_size=batch_size, log_interval=log_interval)
        acc, lo = test(dataloader=test_loader, model=net, loss_fn=criterion, batch_size=batch_size)
        accuracy.append(round(acc, 2))
        loss.append(lo)
    
    return accuracy, loss

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


def test(model: nn.Module, dataloader: DataLoader, loss_fn: nn.CrossEntropyLoss, batch_size: int) -> tuple[float, float]:
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
    return 100*correct, test_loss


def execute_model_complete(learning_rate: float, batch_size: int, n_epochs: int, log_interval: int = 3200, load_dataset: bool = True):

    dataset = create_dataset(load=load_dataset, complete=True)

    train_path = 'datasets/train_dataset_complete.pickle'
    test_path = 'datasets/test_dataset_complete.pickle'

    if load_dataset and os.path.exists(train_path) and os.path.exists(test_path):
        train_loader, test_loader = get_loaders(train_path, test_path)
    else:
        train_loader, test_loader = make_loaders(dataset, batch_size=batch_size, train_path=train_path, test_path=test_path)
    
    accuracy, loss = execute_model(train_loader, test_loader, learning_rate, batch_size, n_epochs, log_interval)
    log(train_dataset="complete", test_dataset="complete", learning_rate=learning_rate, batch_size=batch_size, epochs=n_epochs, accuracy=accuracy, loss=loss)

def execute_model_clean_clean(learning_rate: float, batch_size: int, n_epochs: int, log_interval: int = 3200, load_dataset: bool = True):

    clean, _ = create_dataset(load=load_dataset, complete=False)

    train_path = 'datasets/train_dataset_clean.pickle'
    test_path = 'datasets/test_dataset_clean.pickle'

    if load_dataset and os.path.exists(train_path) and os.path.exists(test_path):
        train_loader, test_loader = get_loaders(train_path, test_path)
    else:
        train_loader, test_loader = make_loaders(clean, batch_size, train_path, test_path)
    
    accuracy, loss = execute_model(train_loader, test_loader, learning_rate, batch_size, n_epochs, log_interval)
    log(train_dataset="clean", test_dataset="clean", learning_rate=learning_rate, batch_size=batch_size, epochs=n_epochs, accuracy=accuracy, loss=loss)


def execute_model_noisy_noisy(learning_rate: float, batch_size: int, n_epochs: int, log_interval: int = 3200, load_dataset: bool = True):

    _, noisy = create_dataset(load=load_dataset, complete=False)

    train_path = 'datasets/train_dataset_noisy.pickle'
    test_path = 'datasets/test_dataset_noisy.pickle'

    if load_dataset and os.path.exists(train_path) and os.path.exists(test_path):
        train_loader, test_loader = get_loaders(train_path, test_path)
    else:
        train_loader, test_loader = make_loaders(noisy, batch_size, train_path, test_path)
    
    accuracy, loss = execute_model(train_loader, test_loader, learning_rate, batch_size, n_epochs, log_interval)
    log(train_dataset="noisy", test_dataset="noisy", learning_rate=learning_rate, batch_size=batch_size, epochs=n_epochs, accuracy=accuracy, loss=loss)

