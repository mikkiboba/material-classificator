
# ? Da discutere per il meet
# ? 1. modificato completamente il codice, devo controllare se i risultati sono giusti
    # ? 1.1. non ho modificato i file di test
    # ?     ho solo aggiunto dei controlli che mi dicono a in quale file e in quale linea vengono trovati gli errori
    # ? 1.2. controllare se i nuovi grafici siano corretti
# ? 2. ho fatto qualche ricerca sulle classificazioni dei segnali
    # ? 2.1. c'è poca roba pubblicata, molte cose sono su youtube
    # ? 2.2. i modelli più utilizzati sono le RNN, LTSM (e combinazioni di esse)
    # ? 2.3. chiedere informazioni sul random forest, possibile utilizzarlo?


# 0 - Imports
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import os

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

def error(name: str, file: str, *args):
    ret = f"{name:15}: {file:25} "
    for i in args:
        ret += f'{i:<15}'
    print(ret)

def fix_length(lista1:list,lista2:list,same:bool=False) -> tuple:
    """
    Fix the len of elements inside the lists.
    If `same` has value `True` then the two lists will have the same amount of elements.
    """
    
    l1 = lista1
    l2 = lista2
        
    min_val1 = len(l1[0])
    for i in range(1,len(l1)):
        if len(l1[i]) < min_val1:
            min_val1 = len(l1[i])
    min_val2 = len(l2[0])
    for i in range(1,len(l2)):
        if len(l2[i]) < min_val2:
            min_val2 = len(l2[i])
        
    if same:
        min_val = min(min_val1,min_val2)
        for i in range(len(l1 if min_val1 == min_val else l2)):
            l1[i] = l1[i][:min_val]
            l2[i] = l2[i][:min_val]
        return l1,l2

    for i in range(len(l1)):
        l1[i] = l1[i][:min_val1]
    for i in range(len(l2)):
        l2[i] = l2[i][:min_val2]
    return l1,l2

def plot_single(materials:list,dataloader:DataLoader):
    """
    Plot the materials' data one by one (clean and noisy)
    """
    
    PLOT_PATH = 'plots/new/singolo'
    for clean_data, noisy_data, labels_dataset in dataloader:
            for i in range(len(labels_dataset)):
                list_clean = clean_data[i].detach().numpy()
                for el in list_clean:
                    plt.plot(el)
                plt.xlabel('subcarrier')
                plt.ylabel('amplitude')
                plt.title(f'{materials[labels_dataset[i]]} CLEAN')
                plt.savefig(f'{PLOT_PATH}/c{i}')
                plt.close()
            for i in range(len(labels_dataset)):
                list_noisy = noisy_data[i].detach().numpy()
                for el in list_noisy:
                    plt.plot(el)
                plt.xlabel('subcarrier')
                plt.ylabel('amplitude')
                plt.title(f'{materials[labels_dataset[i]]} NOISY')
                plt.savefig(f'{PLOT_PATH}/n{i}')
                plt.close()  

def plot_both(materials:list,dataloader:DataLoader):
    """
    Plot the materials' data putting together clean and noisy in the same plot.
    """
    
    PLOT_PATH = 'plots/new/both'
    for clean_data, noisy_data, labels_dataset in dataloader:
        for i in range(len(labels_dataset)):
            figure, axis = plt.subplots(1,2)
            figure.set_size_inches(10,5)
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
            plt.savefig(f'{PLOT_PATH}/cn{i}')
            plt.close() 

def create_dataset(plot:bool=False,both:bool=False,dlen=True,load=True):
    PATH_DATASET = 'dataset_file.pickle'
    materials = [ "empty","acrylic","aluminium","brass","copper","lignum_vitae","nylon","oak","pine","pp","pvc","rose_wood","steel"]
    
    if os.path.exists(PATH_DATASET) and load:
        with open(PATH_DATASET, 'rb') as file:
            dataloader = pickle.load(file)
        print(f'-- DATASET LOADED')
        
    else:
        tests = ["test1.csv","test2.csv","test3.csv","test4.csv","test5.csv","test6.csv","test7.csv","test8.csv","test9.csv","test10.csv"]
        clean_dataset = []
        noisy_dataset=[]
        all_labels=[]

        print(f'-- DATASET CREATION...')
        for i,m in enumerate(materials):
                clean,noisy = [],[]
                all_labels.append(i) 
                for t in tests:
                    # 1.2.1 - Add object label to list of all labels
                    #all_labels.append(i) # ! fuori dal for, aggiunto prima
                    # 1.2.2 - Case 1) Clean data
                    clean_path = "complete_dataset/" + str(m) + "/with/" + str(t)
                    #clean = [] # ! fuori dal for, aggiunto prima
                    with open(clean_path, 'r') as file:
                        for idx,line in enumerate(file):
                            try:
                                values = line.strip().split(',')[25]
                            except:
                                #print(f'{"COLUMN ERROR:":<15} in {str(m)}/with/{str(t)} line {idx+1}') #* error if it can't find the 25th column
                                error("COLUMN ERROR", f'{str(m)}/with/{str(t)}', f'line {idx+1}')
                                continue
                            values = values[1:len(values)-2].split(' ')
                            row = []
                            for i,v in enumerate(values):
                                try:
                                    row.append(float(v))
                                except:
                                    #print(f'{"VALUE ERROR:":<15} value "{v}" in {str(m)}/with/{str(t)} line {idx+1} {"":<5} -> position: {i}') #* error if the value can't be casted to float
                                    error("VALUE  ERROR", f'{str(m)}/with/{str(t)}', f'line {idx+1}', f'value: "{v}"', f'position: {i}')
                            if len(row) == 128:
                                clean.append(row)
                            else:
                                #print(f'{"LENGTH ERROR:":<15} in {str(m)}/with/{str(t)} line {idx+1} {"":<5} -> actual length: {len(row)}') #* error if the number of values is different than 12
                                error("LENGTH ERROR", f'{str(m)}/with/{str(t)}', f'line {idx+1}', f'length: {len(row)}')
                    #clean=torch.tensor(clean) #! fuori dal for, aggiunto dopo
                    #clean_dataset.append(clean) #! fuori dal for, aggiunto dopo
                    # 1.2.3 - Case 2) Noisy data
                    noisy_path = "complete_dataset/" + str(m) + "/without/" + str(t)
                    #noisy = [] #! fuori dal for, aggiunto prima
                    with open(noisy_path, 'r') as file:
                        for idx,line in enumerate(file):
                            try:
                                values = line.strip().split(',')[25]
                            except:
                                #print(f'{"COLUMN ERROR:":<15}in {str(m)}/without/{str(t)} line {idx+1}') #* error if it can't find the 25th column
                                error("COLUMN ERROR", f'{str(m)}/with/{str(t)}', f'line {idx+1}')
                            values = values[1:len(values)-2].split(' ')
                            row = []
                            for i,v in enumerate(values):
                                try:
                                    row.append(float(v))
                                except:
                                    #print(f'{"VALUE ERROR:":<15} value "{v}" in {str(m)}/without/{str(t)} line {idx+1} {"":<5} -> position: {i}') #* error if the value can't be casted to float
                                    error("VALUE  ERROR", f'{str(m)}/with/{str(t)}', f'line {idx+1}', f'value: "{v}"', f'position: {i}')
                            if len(row) == 128:
                                noisy.append(row)
                            else:
                                #print(f'{"LENGTH ERROR:":<15} in {str(m)}/with/{str(t)} line {idx+1} {"":<5} -> actual length: {len(row)}') #* error if the number of values is different than 128
                                error("LENGTH ERROR", f'{str(m)}/with/{str(t)}', f'line {idx+1}', f'length: {len(row)}')
                    #noisy = torch.tensor(noisy) #! fuori dal for, aggiunto dopo
                    #noisy_dataset.append(noisy) #! fuori dal for, aggiunto dopo
                #! mettendoli qui si avrà che, quando verrà fatto torch.stack() avranno shape di [13,14432,128] (esempio)
                clean = torch.tensor(clean)
                clean_dataset.append(clean)
                noisy = torch.tensor(noisy)
                noisy_dataset.append(noisy)
        # 1.3 - Compose clean and noisy dataset
        clean_dataset, noisy_dataset = fix_length(clean_dataset,noisy_dataset,same=dlen) #! torch.stack richeide che le liste dentro clean_dataset e noisy_dataset debbano avere lo stesso numero di elementi  
        clean_dataset=torch.stack(clean_dataset)
        noisy_dataset=torch.stack(noisy_dataset)
        #clean_dataset = torch.permute(clean_dataset, (0,2,1))
        #noisy_dataset = torch.permute(noisy_dataset, (0,2,1))

        #! In questo modo, clean_dataset e noisy_dataset avranno 14000+ tensor per ogni materiale

        # 1.4 - Define Dataset Class
        
        # 1.5 - Build Dataset with Signal Dataset Class
        dataset = SignalDataset(clean_dataset, noisy_dataset, all_labels)
        # 1.6 - Define Parameters
        batch_size = 15
        # 1.7 - Create Dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f'-- DATASET CREATED')
        with open(PATH_DATASET, 'wb') as file:
            pickle.dump(dataloader,file)
 
    if not plot:
        return dataloader
    
    print(f'-- PLOTTING...')
    plot_both(materials,dataloader) if both else plot_single(materials,dataloader)
    print(f'-- PLOTTING DONE')
    
    return dataloader


