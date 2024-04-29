import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset

from net import Cnn, Cnn2
from dataset_creation import create_dataset

from debug import debug
import json


def log(learning_rate: float, batch_size: int, epochs: int, accuracy: list[float], loss: list[float],
        train_dataset: str = "", test_dataset: str = "") -> None:
    log_path = 'results.json'
        
    instance = {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "accuracy": accuracy,
        "max_accuracy": max(accuracy),
        "loss": loss
    }
    
    with open(log_path, 'r+') as file:
        data = json.load(file)
        data['training'].append(instance)
        file.seek(0)
        json.dump(data, file, indent=4)


def train(dataloader: DataLoader, model: nn.Module, loss_fn: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer,
          batch_size: int, log_interval: int = 3200) -> None:
    model.train()
    size = len(dataloader.dataset)
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


def run_model(train_loader: DataLoader, test_loader: DataLoader, learning_rate: float, batch_size: int, n_epochs: int,
              log_interval: int) -> tuple[list[float], list[float]]:
    net = Cnn2()
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    accuracy, loss = [], []
    for epoch in range(n_epochs):
        print(f'Epoch: {epoch+1}\n')
        train(dataloader=train_loader, model=net, loss_fn=criterion, optimizer=optimizer, batch_size=batch_size,
              log_interval=log_interval)
        acc, lo = test(dataloader=test_loader, model=net, loss_fn=criterion, batch_size=batch_size)
        accuracy.append(round(acc, 2))
        loss.append(lo)
    
    path_model = 'model.pth'
    torch.save(net.state_dict(), path_model)
    
    return accuracy, loss


def execute_model(train_name: str, test_name: str, learning_rate: float, batch_size: int, n_epochs: int,
                  log_interval: int = 3200, load_dataset: bool = True) -> None:
    train_dataset = create_dataset(train_name, load_dataset)
    if train_name.lower() == test_name.lower():
        train_dataset, test_dataset = random_split(train_dataset, [0.8, 0.2])
    else:
        test_dataset = create_dataset(test_name, load_dataset if train_name.lower() != 'complete' else True)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    accuracy, loss = run_model(train_loader, test_loader, learning_rate, batch_size, n_epochs, log_interval)
    log(train_dataset=train_name, test_dataset=test_name, learning_rate=learning_rate, batch_size=batch_size,
        epochs=n_epochs, accuracy=accuracy, loss=loss)
