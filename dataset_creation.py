# DATASET CREATION AND VISUALIZATION
# Given clean data files it organizes the dataset and allows for visualization

# 0 - Imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1 - Build Dataset
# 1.1 - Variables to automate the process
materials = [ "empty","acrylic", "pine", "copper", "aluminium" , "brass", "copper", "lignum_vitae", "nylon", "oak", "pine", "pp", "pvc", "rose_wood", "steel"]
shield = ["with", "without"]
tests = ["test1.csv","test2.csv","test3.csv","test4.csv","test5.csv","test6.csv","test7.csv","test8.csv","test9.csv","test10.csv"]
clean_dataset = []
noisy_dataset=[]
all_labels=[] # nb chosen labels: 0(empty), 1(acrylics), 2(pine), 3(copper), 4(aluminium), ...
# 1.2 - Loop to automatically create clean, noisy and labels matrices
for i,m in enumerate(materials):
        for t in tests:
            # 1.2.1 - Add object label to list of all labels
            all_labels.append(i)
            # 1.2.2 - Case 1) Clean data
            clean_path = "complete_dataset/" + str(m) + "/with/" + str(t)
            clean = []
            with open(clean_path, 'r') as file:
                for line in file:
                    values = line.strip().split(',')[25]
                    values = values[1:len(values)-2].split(' ')
                    row = [float(value) for value in values]
                    clean.append(row)
            # Safety print check
            if len(clean) != 1000:
                print(m,"with", t)
            clean=torch.tensor(clean)
            clean_dataset.append(clean)
            # 1.2.3 - Case 2) Noisy data
            noisy_path = "complete_dataset/" + str(m) + "/without/" + str(t)
            noisy = []
            with open(noisy_path, 'r') as file:
                for line in file:
                    values = line.strip().split(',')[25]
                    values = values[1:len(values)-2].split(' ')
                    row = [float(value) for value in values]
                    noisy.append(row)
            # Safety print check
            if len(noisy)!=1000:
                print(m,"without",t)
            noisy = torch.tensor(noisy)
            noisy_dataset.append(noisy)
# 1.3 - Compose clean and noisy dataset
clean_dataset=torch.stack(clean_dataset)
noisy_dataset=torch.stack(noisy_dataset)
# 1.4 - Define Dataset Class
class SignalDataset(Dataset):
    def __init__(self, clean_data, noisy_data, labels):
        # nb: in this way data is paired
        self.clean_data = clean_data
        self.noisy_data = noisy_data
        self.labels = labels

    def __len__(self):
        return len(self.clean_data)

    def __getitem__(self, idx):
        clean_signal = self.clean_data[idx]
        noisy_signal = self.noisy_data[idx]
        label = self.labels[idx]
        return clean_signal, noisy_signal, label

# 1.5 - Build Dataset with Signal Dataset Class
dataset = SignalDataset(clean_dataset, noisy_dataset, all_labels)
# 1.6 - Define Parameters
batch_size = 15
num_epoch = 100
# 1.7 - Create Dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("DATASET CREATED ---------------------------------------------------------------------------")

# 1.8 Visualize clean / noisy data
# NB: lento quindi commentare se non necessario
for clean_data, noisy_data, labels_dataset in dataloader:
      for i in range(len(labels_dataset)):
        print("-----------------------------------------------------------------------------")
        # 1.8.1 - Get label
        label = labels_dataset[i]
        print("$$$$$$$$$$$$$$$$$",str(label),"$$$$$$$$$$$$$$$$$")
         # 1.8.2 - Build list for clean data
        list_clean = clean_data[i].detach().numpy()
        for el in list_clean:
          plt.plot(el)
        plt.xlabel('subcarrier')
        plt.ylabel('amplitude')
        plt.title('CLEAN')
        plt.show()
        plt.savefig('s.png')
        # 1.8.3 - Build list for noisy data
        list_noisy = noisy_data[i].detach().numpy()
        for el in list_noisy:
          plt.plot(el)
        plt.xlabel('subcarrier')
        plt.ylabel('amplitude')
        plt.title('NOISY')
        plt.show()
        plt.savefig('s.png')