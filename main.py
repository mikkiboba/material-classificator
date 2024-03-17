from new_dataset_creation import create_dataset
import time
from net import NNet
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
from debug import debug


def split_tensor(tensor: torch.Tensor, split_ratio: float = 0.8, dimension: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    dim_size = tensor.size(dimension)
    split_size = int(dim_size * split_ratio)

    dataset1 = tensor.narrow(dimension, 0, split_size)
    dataset2 = tensor.narrow(dimension, split_size, dim_size - split_size)

    return dataset1, dataset2


def train(net: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device, log_interval: int) -> None:
    net.train()  # Set the network to training mode
    total_loss = 0
    for batch_idx, (clean, _, labels) in enumerate(train_loader):
        clean, labels = clean.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear the gradients
        output = net(clean)  # Forward pass
        loss = criterion(output, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {batch_idx * len(clean)} / {len(train_loader.dataset)}\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader.dataset)
    print(f'-- TRAINING AVERAGE LOSS: {avg_loss}')


def main() -> None:
    # Parameters
    learning_rate = 0.01
    batch_size = 1
    n_epochs = 100
    log_interval = 10
    path = 'data'
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl = create_dataset(plot=True, load=True)

    net = NNet()

    clean_tensor = torch.Tensor()
    noisy_tensor = torch.Tensor()
    labels_tensor = torch.Tensor()

    for clean, noisy, labels in dl:
        clean_tensor = clean.clone().detach()
        noisy_tensor = noisy.clone().detach()
        labels_tensor = labels.clone().detach()

    clean_train, clean_test = split_tensor(clean_tensor)
    noisy_train, noisy_test = split_tensor(noisy_tensor)

    train_dataset = TensorDataset(clean_train, noisy_train, labels_tensor)
    test_dataset = TensorDataset(clean_test, noisy_test, labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()


if __name__ == "__main__":
    now = time.time()
    main()
    print(f'-- EXECUTION TIME: {time.time()-now} seconds')
