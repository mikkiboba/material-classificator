import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from debug import debug
from sys import exit


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
    Get the signal values from a file.\n
    This has to be changed if the data gets preprocessed.
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


def get_dataset(path: str) -> Dataset | None:
    if os.path.exists(path):
        with open(path, 'rb') as file:
            print(f'-- DATASET LOADED')
            return pickle.load(file)
    return None


def create_dataset_complete(materials: list[str], tests: list[str], dataset_path: str, load: bool) -> Dataset:
    if load:
        return get_dataset(dataset_path)

    clean = create_dataset_single(materials, tests, 'with',
                                  'datasets/clean_dataset_file.pickle', load=load)
    noisy = create_dataset_single(materials, tests, 'without',
                                  'datasets/noisy_dataset_file.pickle', load=load)
    dataset = torch.utils.data.ConcatDataset([clean, noisy])

    with open(dataset_path, 'wb') as file:
        pickle.dump(dataset, file)

    return dataset


def create_dataset_single(materials: list[str], tests: list[str], shield: str, dataset_path: str, load: bool) -> Dataset:
    if load:
        return get_dataset(dataset_path)

    signals, labels = [], []

    for i, m in enumerate(materials):
        for t in tests:
            path = f'complete_dataset/{str(m)}/{shield}/{str(t)}'
            signals, labels = get_values(signals, labels, i, path)
    signals = torch.tensor(signals)

    dataset = SignalDataset(signals, labels)

    with open(dataset_path, 'wb') as file:
        pickle.dump(dataset, file)

    return dataset


def create_dataset(dataset_name: str, load: bool) -> Dataset:
    materials = ["empty", "acrylic", "aluminium", "brass", "copper", "lignum_vitae", "nylon", "oak", "pine", "pp",
                 "pvc", "rose_wood", "steel"]
    tests = ["test1.csv", "test2.csv", "test3.csv", "test4.csv", "test5.csv", "test6.csv", "test7.csv", "test8.csv",
             "test9.csv", "test10.csv"]

    dataset_name = dataset_name.lower()
    path_dataset = f'datasets/{dataset_name}_dataset_file.pickle'
    dataset = None
    if dataset is None:
        print(f'-- {dataset_name.upper()} DATASET CREATION...')
        if dataset_name == 'clean':
            dataset = create_dataset_single(materials, tests, 'with', path_dataset, load)
        elif dataset_name == 'noisy':
            dataset = create_dataset_single(materials, tests, 'without', path_dataset, load)
        elif dataset_name == 'complete':
            dataset = create_dataset_complete(materials, tests, path_dataset, load)
        else:
            raise ValueError(f"dataset_name not in ['clean', 'noisy', 'complete']")
        print(f'-- {dataset_name.upper()} DATASET CREATED.')

    return dataset

