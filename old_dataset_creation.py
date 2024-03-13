
# ! modificato il codice per tenere traccia di tutte le fonti di errore nei file test.csv


# DATASET CREATION AND VISUALIZATION
# Given clean data files it organizes the dataset and allows for visualization

# 0 - Imports
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

plot_folder = "plots"

# 1 - Build Dataset
# 1.1 - Variables to automate the process
materials = ["empty","acrylic"]
#materials = [ "empty", "acrylic", "pine", "copper", "aluminium" , "brass", "lignum_vitae", "nylon", "oak", "pp", "pvc", "rose_wood", "steel"]
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
            clean = [] # ! per ogni test svuota clean, mentre dovrebbe essere svuotato per ogni materiale
            if len(clean) != 1000:
                print("starting", m,"with", t)
            with open(clean_path, 'r') as file:
                for idx,line in enumerate(file):
                    try:
                        values = line.strip().split(',')[25]
                    except:
                        print(f'ERROR: column 25 does not exist - file {clean_path} - line {idx+1} -> IGNORED')
                        break   
                    values = values[1:len(values)-2].split(' ')
                    row = []                        
                    for v in values:
                        try:
                            if v != '':
                                row.append(float(v)) # !!!!!! row si annulla ad ogni iterazione senza salvare niente. Si salva soltanto l'ultima riga 
                        except:
                            print(f'ERRORE: values error - file {clean_path} - line {idx+1} -> IGNORED')
                            print(f'value: {v}')
            if len(row) == 128:
                clean.append(row)
            else:
                print(f'ERRORE: row length - {str(m)}/with/{str(t)}')
            # Safety print check
            
            clean=torch.tensor(clean)
            clean_dataset.append(clean)
            # 1.2.3 - Case 2) Noisy data
            noisy_path = "complete_dataset/" + str(m) + "/without/" + str(t)
            noisy = []
            if len(clean) != 1000:
                print("starting",m,"without", t)
            with open(noisy_path, 'r') as file:
                for idx,line in enumerate(file):
                    try:
                        values = line.strip().split(',')[25]
                    except:
                        print(f'ERROR: column 25 does not exist - file {noisy_path} - line {idx+1}')
                        break
                    values = values[1:len(values)-2].split(' ')
                    row = []                        
                    for v in values:
                        try:
                            if v != '':
                                row.append(float(v))
                        except:
                            print(f'ERRORE: value error - file {noisy_path} - line {idx+1}')
                            print(f'value: {v}')
            if len(row) == 128:
                noisy.append(row)
            else:
                print(f'ERRORE: row length - {str(m)}/without/{str(t)}')
            # Safety print check
            noisy = torch.tensor(noisy)
            noisy_dataset.append(noisy)
# 1.3 - Compose clean and noisy dataset
clean_dataset=torch.stack(clean_dataset)
noisy_dataset=torch.stack(noisy_dataset)
print(clean_dataset.shape)

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
# 1.7 - Create Dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("DATASET CREATED ---------------------------------------------------------------------------")
'''
# 1.8 Visualize clean / noisy data
# NB: lento quindi commentare se non necessario
for clean_data, noisy_data, labels_dataset in dataloader:
      for i in range(len(labels_dataset)):
        
        print("-----------------------------------------------------------------------------")
        # 1.8.1 - Get label
        label = labels_dataset[i]
        print("$$$$$$$$$$$$$$$$$",str(label),"$$$$$$$$$$$$$$$$$")
         # 1.8.2 - Build list for clean data
        list_clean = clean_data[i].detach().numpy() #! qui clean_data[i] ha un solo elemento
        for el in list_clean:
          plt.plot(el)
        plt.xlabel('subcarrier')
        plt.ylabel('amplitude')
        plt.title('CLEAN')
        #plt.show()
        plt.savefig(f'{plot_folder}/old/c{i}.png')
        # 1.8.3 - Build list for noisy data
        list_noisy = noisy_data[i].detach().numpy()
        for el in list_noisy:
          plt.plot(el)
        plt.xlabel('subcarrier')
        plt.ylabel('amplitude')
        plt.title('NOISY')
        #plt.show()
        plt.savefig(f'{plot_folder}/old/n{i}.png')
        '''
            