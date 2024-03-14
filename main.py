

def debug(*args):
    """
    Debug print
    """
    ret = ""
    for i in args:
        ret += str(i) + " "
    print(f'-- DEBUG {ret}')





def main():
    now = time.time()

    dl = create_dataset(
        plot=False,
        load=True)
    
    net = Net()

    clean_tensor = torch.Tensor()
    noisy_tensor = torch.Tensor()
    labels_tensor = torch.Tensor()

    for clean, noisy, labels in dl:
        clean_tensor = clean.clone().detach()
        noisy_tensor = noisy.clone().detach()
        labels_tensor = labels.clone().detach()
    dataset = TensorDataset(clean_tensor, noisy_tensor, labels_tensor)
    
    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[len(dataset) - int(len(dataset)*.2), int(len(dataset)*.2)]
    )
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=.001)
    
    batch_size = 1
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    num_epochs = 5
    
    
    for epoch in range(num_epochs):  # num_epochs is defined by you
        net.train()
        running_loss = 0.0
        for batch_idx, (inputs, _, labels) in enumerate(train_loader):
            debug(inputs.shape)
            optimizer.zero_grad()  # Zero the gradients
            
            outputs = net(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize
            
            running_loss += loss.item()  # Accumulate loss
            
            print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {running_loss / (batch_idx+1)}')

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, _, labels) in enumerate(test_loader):
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the test set: {100 * correct / total}%')
    print(f'-- EXECUTION TIME: {time.time()-now} seconds')


if __name__ == "__main__":
    from new_dataset_creation import create_dataset
    import time
    from net import Net
    import torch.optim as optim
    import torch.nn as nn
    from torch.utils.data import random_split,DataLoader,TensorDataset
    import torch
    main()
