
#TODO: stesso training ma solo su clean e noisy
#TODO: controllare i giusti valori da mettere nei conv layers
#TODO: decidere ufficialmente come e quali valori cambiare per i vari test

import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset

from net import Cnn, Cnn2
from dataset_creation import create_dataset

from debug import debug
import sys
import os
import json


def log(learning_rate: float, batch_size: int, epochs: int, accuracy: list[float], loss: list[float], fft: bool, train_dataset: str = "", test_dataset: str = ""):
    log_path = 'results.json'
        
    instance = {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "fft": fft,
        "accuracy": accuracy,
        "max_accuracy": max(accuracy),
        "loss": loss
    }
    
    with open(log_path, 'r+') as file:
        data = json.load(file)
        data['training'].append(instance)
        file.seek(0)
        json.dump(data, file, indent=4)


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


def train(dataloader: DataLoader, model: nn.Module, loss_fn: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer,
          batch_size: int, log_interval: int = 3200):

    size = len(dataloader.dataset)
    model.train()

    data: torch.Tensor

    for batch_idx, (data, target) in enumerate(dataloader):
        if data.size(0) != batch_size or target.size(0) != batch_size:
            continue
        pred = model(data)
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
            #pred = model(data.unsqueeze(1))
            pred = model(data)
            target = target.long()
            test_loss += loss_fn(pred, target).item()
            correct += (pred.argmax(1) == target).float().sum().item()

    test_loss /= num_batches
    correct /= size
    print(f'Test error: \n Accuracy: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f} \n')
    return 100*correct, test_loss


def run_model(train_loader: DataLoader, test_loader: DataLoader, learning_rate: float, batch_size: int, n_epochs: int, log_interval: int, fft: bool) -> tuple[list[float], list[float]]:
    net = Cnn2(fft=fft)
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    accuracy, loss = [], []
    for epoch in range(n_epochs):
        print(f'Epoch: {epoch+1}\n')
        train(dataloader=train_loader, model=net, loss_fn=criterion, optimizer=optimizer, batch_size=batch_size, log_interval=log_interval)
        acc, lo = test(dataloader=test_loader, model=net, loss_fn=criterion, batch_size=batch_size)
        accuracy.append(round(acc, 2))
        loss.append(lo)
    
    path_model = 'model.pth'
    torch.save(net.state_dict(), path_model)
    
    return accuracy, loss


'''
def execute_model(train_dataset: str, test_dataset: str, learning_rate: float, batch_size: int, n_epochs: int, log_interval: int = 3200, load_dataset: bool = True, fft: bool = True):
    train_path = f'datasets/train_dataset_{train_dataset.lower()}.pickle'
    test_path = f'datasets/test_dataset_{test_dataset.lower()}.pickle'
    
    def find_dataset(dataset_name: str) -> Dataset:
        if dataset_name == 'clean':
            dataset, _ = create_dataset(load=load_dataset, complete=False, fft=fft)
        elif dataset_name == 'noisy':
            _, dataset = create_dataset(load=load_dataset, complete=False, fft=fft)
        else:
            dataset = create_dataset(load=load_dataset, complete=True, fft=fft)
        return dataset
    
    if load_dataset and os.path.exists(train_path) and os.path.exists(test_path):
        train_loader, test_loader = get_loaders(train_path, test_path)
    else:
        tr_dataset = find_dataset(train_dataset)
        te_dataset = find_dataset(test_dataset)
        
        train_loader, _ = make_loaders(tr_dataset, batch_size=batch_size, train_path=train_path)
'''        
    

def execute_model_complete(learning_rate: float, batch_size: int, n_epochs: int, log_interval: int = 3200, load_dataset: bool = True, fft: bool = True):

    dataset = create_dataset(load=load_dataset, complete=True, fft=fft)

    train_path = 'datasets/train_dataset_complete.pickle'
    test_path = 'datasets/test_dataset_complete.pickle'

    if load_dataset and os.path.exists(train_path) and os.path.exists(test_path):
        train_loader, test_loader = get_loaders(train_path, test_path)
    else:
        train_loader, test_loader = make_loaders(dataset, batch_size=batch_size, train_path=train_path, test_path=test_path)
    
    accuracy, loss = run_model(train_loader, test_loader, learning_rate, batch_size, n_epochs, log_interval, fft)
    log(train_dataset="complete", test_dataset="complete", learning_rate=learning_rate, batch_size=batch_size, epochs=n_epochs, accuracy=accuracy, loss=loss, fft=fft)


def execute_model_clean_clean(learning_rate: float, batch_size: int, n_epochs: int, log_interval: int = 3200, load_dataset: bool = True, fft: bool = True):

    clean, _ = create_dataset(load=load_dataset, complete=False, fft=fft)

    train_path = 'datasets/train_dataset_clean.pickle'
    test_path = 'datasets/test_dataset_clean.pickle'

    if load_dataset and os.path.exists(train_path) and os.path.exists(test_path):
        train_loader, test_loader = get_loaders(train_path, test_path)
    else:
        train_loader, test_loader = make_loaders(clean, batch_size, train_path, test_path)
    
    accuracy, loss = run_model(train_loader, test_loader, learning_rate, batch_size, n_epochs, log_interval, fft)
    log(train_dataset="clean", test_dataset="clean", learning_rate=learning_rate, batch_size=batch_size, epochs=n_epochs, accuracy=accuracy, loss=loss, fft=fft)


def execute_model_noisy_noisy(learning_rate: float, batch_size: int, n_epochs: int, log_interval: int = 3200, load_dataset: bool = True, fft: bool = True):

    _, noisy = create_dataset(load=load_dataset, complete=False, fft=True)

    train_path = 'datasets/train_dataset_noisy.pickle'
    test_path = 'datasets/test_dataset_noisy.pickle'

    if load_dataset and os.path.exists(train_path) and os.path.exists(test_path):
        train_loader, test_loader = get_loaders(train_path, test_path)
    else:
        train_loader, test_loader = make_loaders(noisy, batch_size, train_path, test_path)
    
    accuracy, loss = run_model(train_loader, test_loader, learning_rate, batch_size, n_epochs, log_interval, fft)
    log(train_dataset="noisy", test_dataset="noisy", learning_rate=learning_rate, batch_size=batch_size, epochs=n_epochs, accuracy=accuracy, loss=loss, fft=fft)
    

def execute_model_clean_noisy(learning_rate: float, batch_size: int, n_epochs: int, log_interval: int = 3200, load_dataset: bool = True, fft: bool = True):
    clean, noisy = create_dataset(load=load_dataset, complete=False, fft=fft)
    
    clean_loader = DataLoader(clean, batch_size=batch_size, shuffle=True)
    noisy_loader = DataLoader(noisy, batch_size=batch_size, shuffle=True)
    
    accuracy, loss = run_model(train_loader=clean_loader, test_loader=noisy_loader, learning_rate=learning_rate, batch_size=batch_size, n_epochs=n_epochs, log_interval=log_interval, fft=fft)
    log(train_dataset="clean", test_dataset="noisy", learning_rate=learning_rate, batch_size=batch_size, epochs=n_epochs, accuracy=accuracy, loss=loss, fft=fft)


def execute_model_noisy_clean(learning_rate: float, batch_size: int, n_epochs: int, log_interval: int = 3200, load_dataset: bool = True, fft: bool = True):
    clean, noisy = create_dataset(load=load_dataset, complete=False, fft=fft)
    
    clean_loader = DataLoader(clean, batch_size=batch_size, shuffle=True)
    noisy_loader = DataLoader(noisy, batch_size=batch_size, shuffle=True)
    
    accuracy, loss = run_model(train_loader=noisy_loader, test_loader=clean_loader, learning_rate=learning_rate, batch_size=batch_size, n_epochs=n_epochs, log_interval=log_interval, fft=fft)
    log(train_dataset="noisy", test_dataset="clean", learning_rate=learning_rate, batch_size=batch_size, epochs=n_epochs, accuracy=accuracy, loss=loss, fft=fft)


def execute_model_complete_clean(learning_rate: float, batch_size: int, n_epochs: int, log_interval: int = 3200, load_dataset: bool = True, fft: bool = True):
    complete = create_dataset(load=load_dataset, complete=True, fft=fft)
    clean, _ = create_dataset(load=True, complete=False, fft=fft)
    
    complete_loader = DataLoader(complete, batch_size=batch_size, shuffle=True)    
    clean_loader = DataLoader(clean, batch_size=batch_size, shuffle=True)
    
    accuracy, loss = run_model(train_loader=complete_loader, test_loader=clean_loader, learning_rate=learning_rate, batch_size=batch_size, n_epochs=n_epochs, log_interval=log_interval, fft=fft)
    log(train_dataset="complete", test_dataset="clean", learning_rate=learning_rate, batch_size=batch_size, epochs=n_epochs, accuracy=accuracy, loss=loss, fft=fft)


def execute_model_complete_noisy(learning_rate: float, batch_size: int, n_epochs: int, log_interval: int = 3200, load_dataset: bool = True, fft: bool = True):
    complete = create_dataset(load=load_dataset, complete=True, fft=fft)
    _, noisy = create_dataset(load=True, complete=False, fft=fft)
    
    complete_loader = DataLoader(complete, batch_size=batch_size, shuffle=True)    
    noisy_loader = DataLoader(noisy, batch_size=batch_size, shuffle=True)
    
    accuracy, loss = run_model(train_loader=complete_loader, test_loader=noisy_loader, learning_rate=learning_rate, batch_size=batch_size, n_epochs=n_epochs, log_interval=log_interval, fft=fft)
    log(train_dataset="complete", test_dataset="noisy", learning_rate=learning_rate, batch_size=batch_size, epochs=n_epochs, accuracy=accuracy, loss=loss, fft=fft)
