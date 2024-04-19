import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
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
                    error("VALUE  ERROR", f'{file_name}', f'line {idx+1}', f'value: "{v}"')
            if len(row) == 128:
                collector.append(row)
                labels.append(material_idx)
            else:
                error("LENGTH ERROR", f'{file_name}', f'line {idx+1}', f'length: {len(row)}')
    return collector, labels

def get_dataset(path: str, load: bool) -> DataLoader | None:
    if load and os.path.exists(path):
        with open(path, 'rb') as file:
            print(f'-- DATASET LOADED')
            return pickle.load(file)
    return None


def create_dataset_separate(materials: list[str], tests: list[str], load: bool, fft: bool) -> tuple[Dataset, Dataset]:
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

        if fft:
            clean = np.abs(np.fft.rfft(clean, axis=1)).astype(np.float32)
            noisy = np.abs(np.fft.rfft(noisy, axis=1)).astype(np.float32)
            clean = torch.from_numpy(clean)
            noisy = torch.from_numpy(noisy)
        else:
            clean = torch.tensor(clean)
            noisy = torch.tensor(noisy)

        clean_dataset = SignalDataset(clean, clean_labels)
        noisy_dataset = SignalDataset(noisy, clean_labels)

        with open(path_dataset_clean, 'wb') as file:
            pickle.dump(clean_dataset, file)
        with open(path_dataset_noisy, 'wb') as file:
            pickle.dump(noisy_dataset, file)

        print(f'-- SEPARATE DATASETS CREATED')

    return clean_dataset, noisy_dataset


def create_dataset_complete(materials: list[str], tests: list[str], load: bool, fft: bool) -> Dataset:
    path_dataset = 'datasets/complete_dataset_file.pickle'
    dataset = get_dataset(path_dataset, load)
    if dataset is None:
        print(f'-- COMPLETE DATASET CREATION...')

        clean, noisy = create_dataset_separate(materials, tests, load, fft)
        dataset = torch.utils.data.ConcatDataset([clean, noisy])

        with open(path_dataset, 'wb') as file:
            pickle.dump(dataset, file)

        print(f'-- COMPLETE DATASET CREATED')

    return dataset


def create_dataset(load: bool = True, complete: bool = True, fft: bool = True) -> Dataset | tuple[Dataset, Dataset]:
    materials = ["empty", "acrylic", "aluminium", "brass", "copper", "lignum_vitae", "nylon", "oak", "pine", "pp",
                 "pvc", "rose_wood", "steel"]
    tests = ["test1.csv", "test2.csv", "test3.csv", "test4.csv", "test5.csv", "test6.csv", "test7.csv", "test8.csv",
             "test9.csv", "test10.csv"]

    if complete:
        return create_dataset_complete(materials, tests, load, fft)
    return create_dataset_separate(materials, tests, load, fft)


