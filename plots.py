import matplotlib.pyplot as plt
import json
from dataset_creation import create_dataset
from torch.utils.data import DataLoader
import os
import numpy as np


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
    materials = [ "empty","acrylic", "pine", "copper", "aluminium" , "brass", "copper", "lignum_vitae", "nylon", "oak", "pine", "pp", "pvc", "rose_wood", "steel"]
    shield = ["with", "without"]
    tests = ["test1.csv","test2.csv","test3.csv","test4.csv","test5.csv","test6.csv","test7.csv","test8.csv","test9.csv","test10.csv"]
    base = 'complete_dataset2'
    plot_path = os.path.join('plots', 'preprocessed')
    os.makedirs(plot_path, exist_ok=True)

    for m in materials:
        for s in shield:
            for i, t in enumerate(tests):
                path = os.path.join(base, m, s, t)
                with open(path, 'r') as file:
                    for line in file:
                        values = [float(v) for v in line.split(' ')]
                        plt.plot(values)
            plt.savefig(f'{plot_path}/{m}{s.capitalize()}{i}.png')
            plt.close()


def plot_signals_gpt():
    materials = [ "empty","acrylic", "pine", "copper", "aluminium" , "brass", "copper", "lignum_vitae", "nylon", "oak", "pine", "pp", "pvc", "rose_wood", "steel"]
    shield = ["with", "without"]
    tests = ["test1.csv","test2.csv","test3.csv","test4.csv","test5.csv","test6.csv","test7.csv","test8.csv","test9.csv","test10.csv"]
    base = 'complete_dataset2'
    plot_path = os.path.join('plots', 'preprocessed')
    os.makedirs(plot_path, exist_ok=True)
    
    for m in materials:
        for s in shield:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            ax1.set_title('CSI Amplitude')
            ax2.set_title('CSI Phase')
            ax1.set_ylabel('Amplitude')
            ax2.set_ylabel('Phase (radians)')
            ax2.set_xlabel('Subcarrier Index')
            for i, t in enumerate(tests):
                csv_file = os.path.join(base, m, s, t)
                with open(csv_file, 'r') as file:
                    for line in file:
                        values = np.array(line.split(), dtype=float)
                        num_values = len(values) // 2
                        amplitudes = values[:num_values]
                        phases = values[num_values:]
                        ax1.plot(amplitudes, label=f'Signal {i+1}')
                        ax2.plot(phases, label=f'Signal {i+1}')
            plt.tight_layout()
            plt.savefig(f'plots/preprocessed/{m}{s.capitalize()}')
            plt.close()


plot_signals_gpt()