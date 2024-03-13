# GENERATE DENOISED DATA AND VISUALIZE
# Given the model and noisy data it generates clean data and allows for visualization

# 0 - Imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, dataloader
import torch.autograd as autograd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1 -  Generate denoised signal
denoised_dataset = []
denoised_labels = []
svm_clean_dataset =[]
svm_clean_labels =[]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for clean_data, noisy_data, labels_dataset in dataloader:
    for i in range(len(clean_data)):
        # 1.1 - Get label
        label = labels_dataset[i]
        # 1.2 - Build list for SVM clean data
        list_clean = clean_data[i].detach().numpy()
        # 1.2.1 - Visualize
        for el in list_clean:
            plt.plot(el)
        plt.xlabel('subcarrier')
        plt.ylabel('amplitude')
        plt.title('REAL')
        plt.legend()
        plt.show()
        plt.savefig('s.png')
        svm_clean_dataset.append(list_clean)
        svm_clean_labels.append(label)
        # 1.3 - Build list of denoised data
        noisy_signal = noisy_data[i]
        noisy_signal = noisy_signal.cpu()
        noisy_signal = noisy_signal.unsqueeze(0).to(device)
        denoised = ragan.denoise(noisy_signal)
        denoised = denoised[0]
        list_denoised = denoised.detach().cpu().numpy()
        # 1.3.1 - Visualize
        for el in list_denoised:
            plt.plot(el)
        plt.xlabel('subcarrier')
        plt.ylabel('amplitude')
        plt.title('FAKE')
        plt.legend()
        plt.show()
        plt.savefig('v.png')
        denoised_dataset.append(list_denoised)
        denoised_labels.append(label)

print("DENOISED DATASET CREATED  ---------------------")


