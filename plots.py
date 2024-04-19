import matplotlib.pyplot as plt
import json
from last_dataset_creation import create_dataset
from torch.utils.data import DataLoader


def load(tag: str):
    
    accuracies = []
    with open('results.json', 'r+') as file:
        data = json.load(file)
        
    for instance in data['training']:
        accuracies.append(instance[f'{tag}'])
    return accuracies


def plot_results():
    acc = load('accuracy')
    los = load('loss')
    for a in acc:
        plt.plot(a)
    plt.savefig('plots/results.png')
    plt.close()
    for l in los:
        plt.plot(l)
    plt.savefig('plots/losses.png')
    
    
def plot_signals():
    materials = ["empty", "acrylic", "aluminium", "brass", "copper", "lignum_vitae", "nylon", "oak", "pine", "pp",
                 "pvc", "rose_wood", "steel"]
    clean, _ = create_dataset(load=False, complete=False, fft=False)
    loader = DataLoader(clean, batch_size=64, shuffle=False)
    signals_by_label = {i: [] for i in range(13)}

    # Collect signals for each label
    for signals, labels in loader:
        for signal, label in zip(signals, labels):
            signals_by_label[label.item()].append(signal.squeeze())  # Adjust depending on the shape of your signal

    # Function to plot and save figures for each label
    def plot_signals_for_label(label, signals):
        plt.figure(figsize=(10, 5))  # Adjust the size as needed
        for signal in signals:
            plt.plot(signal.numpy(), label=f'Signal for Label {materials[label]}')  # Adjust if the signal needs more preprocessing
        plt.title(f'Signals for Label {materials[label]}')
        plt.savefig(f'plots/last/clean-{materials[label]}.png')  # Save each plot to a file
        plt.close()  # Close the figure to free up memory
        # Generate and save plots for each label
    for label, signals in signals_by_label.items():
        plot_signals_for_label(label, signals)
plot_signals()