import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from net import NNet
from new_dataset_creation import create_dataset

from debug import debug
import sys


def split_tensor(tensor: torch.Tensor, split_ratio: float = 0.8, dimension: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    dim_size = tensor.size(dimension)
    split_size = int(dim_size * split_ratio)

    dataset1 = tensor.narrow(dimension, 0, split_size)
    dataset2 = tensor.narrow(dimension, split_size, dim_size - split_size)

    return dataset1, dataset2

#TODO: Devo fare le funzioni di training

def train(dataloader: DataLoader, model: NNet, loss_fn: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer, batch_size: int, log_interval: int = 1):
    size = len(dataloader.dataset)
    model.train()
    
    for batch_idx, (data, _, target) in enumerate(dataloader):
        pred = model(data)
        loss = loss_fn(pred, target)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        if batch_idx % log_interval == 0:
            loss, current = loss.item(), batch_idx * batch_size + len(data)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')


def test(model: nn.Module, dataloader: DataLoader, loss_fn: nn.CrossEntropyLoss) -> None:
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for data, _, target in dataloader:
            pred = model(data)
            test_loss += loss_fn(pred, target).item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    
    print(f'Test error: \n Accuracy: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f} \n')


def main() -> None:
    learning_rate = 0.0001
    batch_size = 1
    n_epochs = 10
    log_interval = 1
    path = 'data'
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl = create_dataset(plot=False, load=False)
    
    # unreavle

    clean_tensor = torch.Tensor()
    noisy_tensor = torch.Tensor()
    labels_tensor = torch.Tensor()

    for clean, noisy, labels in dl:
        clean_tensor = clean.clone().detach()
        noisy_tensor = noisy.clone().detach()
        labels_tensor = labels.clone().detach()
        
    clean_train, clean_test = split_tensor(clean_tensor)
    noisy_train, noisy_test = split_tensor(noisy_tensor)
    net = NNet(clean_train.size(1), clean_test.size(1)) #? 1DCnn

    train_dataset = TensorDataset(clean_train, noisy_train, labels_tensor)
    test_dataset = TensorDataset(clean_test, noisy_test, labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    #TODO: Qui dovrebbe andarci training e test
    #? come si deve fare il train, devo fare due training diversi per clean e noisy?
    for epoch in range(n_epochs):
        print(f'Epoch: {epoch+1}\n')
        train(dataloader=train_loader, model=net, loss_fn=criterion, optimizer=optimizer, batch_size=batch_size, log_interval=log_interval)
        test(dataloader=test_loader, model=net, loss_fn=criterion)


if __name__ == "__main__":
    now = time.time()
    main()
    print(f'-- EXECUTION TIME: {time.time()-now} seconds')
