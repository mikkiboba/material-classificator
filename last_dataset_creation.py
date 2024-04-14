import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import os
from debug import debug
from sys import exit

# TODO: devo restituire 2 tipi di dataset
# TODO: 1. dataset clean e dataset noisy
# TODO: 2. dataset clean + noisy


class SignalDataset(Dataset):
    def __init__(self, clean_data, labels):
        self.clean_data = clean_data
        self.labels = labels

    def __len__(self):
        return len(self.clean_data)

    def __getitem__(self, idx):
        clean_signal = self.clean_data[idx]
        label = self.labels[idx]
        return clean_signal, label


def error(name: str, file: str, *args) -> None:
    ret = f"!- {name:15}: {file:25} "
    for i in args:
        ret += f'{i:<15}'
    print(ret)


def get_values(collector: list, labels: list, material_idx: int, path: str) -> tuple[list, list]:
    """
    Get the signal values from a file
    """
    file_name = path[path.find('/')+1:]
    with open(path, 'r') as file:
        for idx, line in enumerate(file):
            try:
                values = line.strip().split(',')[25]
            except IndexError:
                error("COLUMN ERROR", f'{file_name}', f'line {idx+1}')
                continue
            values = values[1:len(values)-2].split(' ')
            row = []
            for idx_val, v in enumerate(values):
                try:
                    row.append(float(v))
                except ValueError:
                    error("VALUE  ERROR", f'{file_name}', f'line {idx+1}', f'value: "{v}"', f'position: {idx_val}')
            if len(row) == 128:
                collector.append(row)
                labels.append(material_idx)
            else:
                error("LENGTH ERROR", f'{file_name}', f'line {idx+1}', f'length: {len(row)}')
    return collector, labels


def plot_single(materials: list, dataloader: DataLoader) -> None:
    """
    Plot the materials' data one by one (clean and noisy)
    """
    def plot(data, title: str, file_name: str) -> None:
        for i in range(len(labels_dataset)):
            list_clean = data[i].detach().numpy()
            for el in list_clean:
                plt.plot(el)
            plt.xlabel('subcarrier')
            plt.ylabel('amplitude')
            plt.title(f'{materials[labels_dataset[i]]} CLEAN')
            plt.savefig(f'{path}/{file_name}{i}')
            plt.close()

    path = 'plots/new/singolo'
    for clean_data, noisy_data, labels_dataset in dataloader:
        plot(clean_data, "CLEAN", 'c')
        plot(noisy_data, "NOISY", 'n')


def plot_both(materials: list, dataloader: DataLoader) -> None:
    """
    Plot the materials' data putting together clean and noisy in the same plot.
    """
    
    path = 'plots/new/both'
    for clean_data, noisy_data, labels_dataset in dataloader:
        for i in range(len(labels_dataset)):
            figure, axis = plt.subplots(1, 2)
            figure.set_size_inches(10, 5)
            list_clean = clean_data[i].detach().numpy()
            list_noisy = noisy_data[i].detach().numpy()
            for el in list_clean:
                axis[0].plot(el)
            for el in list_noisy:
                axis[1].plot(el)
            axis[0].set_xlabel('subcarrier')
            axis[0].set_ylabel('amplitude')
            axis[0].set_title(f'{materials[labels_dataset[i]]} CLEAN')
            axis[1].set_xlabel('subcarrier')
            axis[1].set_ylabel('amplitude')
            axis[1].set_title(f'{materials[labels_dataset[i]]} NOISY')
            plt.savefig(f'{path}/cn{i}')
            plt.close() 


def get_dataset(path: str, load: bool) -> DataLoader | None:
    if load and os.path.exists(path):
        with open(path, 'rb') as file:
            print(f'-- DATASET LOADED')
            return pickle.load(file)
    return None


def create_dataset_separate(materials: list[str], tests: list[str], load: bool) -> tuple[Dataset, Dataset]:
    path_dataset_clean = 'datasets/clean_dataset_file.pickle'
    path_dataset_noisy = 'datasets/noisy_dataset_file.pickle'
    clean_dataset = get_dataset(path_dataset_clean, load)
    noisy_dataset = get_dataset(path_dataset_noisy, load)

    if clean_dataset is None or noisy_dataset is None:
        print(f'-- SEPARATE DATASET CREATION...')

        clean_labels, noisy_labels = [], []
        clean, noisy = [], []

        for i, m in enumerate(materials):
            for t in tests:
                clean_path = "complete_dataset/" + str(m) + "/with/" + str(t)
                clean, clean_labels = get_values(clean, clean_labels, i, clean_path)
                noisy_path = "complete_dataset/" + str(m) + "/without/" + str(t)
                noisy, noisy_labels = get_values(noisy, noisy_labels, i, noisy_path)

        clean = torch.tensor(clean)
        noisy = torch.tensor(noisy)

        clean_dataset = SignalDataset(clean, clean_labels)
        noisy_dataset = SignalDataset(noisy, clean_labels)

        with open(path_dataset_clean, 'wb') as file:
            pickle.dump(clean_dataset, file)
        with open(path_dataset_noisy, 'wb') as file:
            pickle.dump(noisy_dataset, file)

        print(f'-- DATASET CREATED')

    return clean_dataset, noisy_dataset


def create_dataset_complete(materials: list[str], tests: list[str], load: bool) -> Dataset:
    path_dataset = 'datasets/complete_dataset_file.pickle'
    dataset = get_dataset(path_dataset, load)
    if dataset is None:
        print(f'-- COMPLETE DATASET CREATION...')

        clean, noisy = create_dataset_separate(materials, tests, load)
        dataset = torch.utils.data.ConcatDataset([clean, noisy])

        with open(path_dataset, 'wb') as file:
            pickle.dump(dataset, file)

        print(f'-- DATASET CREATED')

    return dataset


def create_dataset(load: bool = True, complete: bool = True) -> Dataset | tuple[Dataset, Dataset]:
    materials = ["empty", "acrylic", "aluminium", "brass", "copper", "lignum_vitae", "nylon", "oak", "pine", "pp",
                 "pvc", "rose_wood", "steel"]
    tests = ["test1.csv", "test2.csv", "test3.csv", "test4.csv", "test5.csv", "test6.csv", "test7.csv", "test8.csv",
             "test9.csv", "test10.csv"]

    if complete:
        return create_dataset_complete(materials, tests, load)
    return create_dataset_separate(materials, tests, load)


