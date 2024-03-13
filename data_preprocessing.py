# DATA PRE-PROCESSING
# Given a csv file of raw CSI data returns a new file with only CSI amplitude and only 1000 central lines

# 0 - Imports
import re
from math import sqrt, atan2
import matplotlib.pyplot as plt

# 1 - Support arrays to iterate over files
# 1.1 - Reduced array of materials ( i 4 cubi iniziali + vuoto)
materials_red = [ "empty","acrylic", "pine", "copper", "aluminium" ]
# 1.2 - Complete array of materials ( tutti i cubi + vupto)
materials_complete = [ "empty","acrylic", "pine", "copper", "aluminium" , "brass", "copper", "lignum_vitae", "nylon", "oak", "pine", "pp", "pvc", "rose_wood", "steel"]
# 1.3 - Array for shield ( with = dentro scatolo, without = fuori scatolo)
shield = ["with", "without"]
# 1.4 - Complete array for tests ( per test iniziali su reduced dataset)
tests_complete = ["test1.csv","test2.csv","test3.csv","test4.csv","test5.csv","test6.csv","test7.csv","test8.csv","test9.csv","test10.csv","test11.csv","test12.csv","test13.csv","test14.csv","test15.csv","test16.csv","test17.csv","test18.csv","test19.csv","test20.csv","test21.csv","test22.csv","test23.csv","test24.csv","test25.csv","test26.csv","test27.csv","test28.csv","test29.csv","test30.csv"]
# 1.5 - Reduced array for tests ( per test finali su complete dataset)
tests_reduced = ["test1.csv","test2.csv","test3.csv","test4.csv","test5.csv","test6.csv","test7.csv","test8.csv","test9.csv","test10.csv"]

# 2 - Define dataset
materials = materials_complete
shield = shield 
tests = tests_reduced

# 3 - Loop to do the pre-processing for each file
for m in materials: 
    for s in shield:
        for t in tests:
            # 3.1 - Assemble the acquisition path
            acquisition_path = "complete_dataset/"+ str(m)+"/"+ str(s)+"/"+ str(t)
            # 3.2 - Open raw CSI data file
            raw_data = open(acquisition_path)
            # 3.3 - Assemble the pre-processed path
            preprocessed_path = "complete_datasete/" + str(m) + "/" + str(s) + "/" + str(t)
            # 3.4 - Open preprocessed data file in writing mode
            preprocessed_data = open(preprocessed_path, 'w')
            # 3.5 - transform raw CSI out from the ESP32 into CSI-amplitude and CSI-phase
            for j, l in enumerate(raw_data.readlines()): 
                imaginary = []
                real = []
                amplitudes = []
                phases = []
                try:
                    # 3.5.1 - Parse string to create integer list
                    # NB: credo ci sia qualcosa di sbagliato in questo pezzetto perchè ci sono alcuni subcarrier sempre nulli anche
                    #     dopo il pre-processing tra 20esimo e 30esimo subcarrier (credo 26/27)
                    csi_string = re.findall(r"\[(.*)]", l)[0]
                    csi_raw = [int(x) for x in csi_string.split(" ") if x != '']
                    csi_raw = csi_raw[13:-12]
                    csi_raw.pop(53)
                    csi_raw.pop(53)
                    # 3.5.2 - Create list of imaginary and real numbers from CSI
                    for i in range(len(csi_raw)):
                        if i % 2 == 0:
                            imaginary.append(csi_raw[i])
                        else:
                            real.append(csi_raw[i])
                    # 3.5.3 - Transform imaginary and real into amplitude and phase
                    for i in range(int(len(csi_raw) / 2)):
                        amplitudes.append(sqrt(imaginary[i] ** 2 + real[i] ** 2))
                        phases.append(atan2(imaginary[i], real[i]))
                    # NB: A volte si impiccia e prende righe non da 50 quindi ho aggiunto questo check
                    if len(amplitudes)!=50:
                        continue
                    preprocessed_data.write(' '.join([f'{el}' for el in amplitudes]) + '\n')

                #NB: questo serve perchè ci sono alcune anomalie nei dati (a volte righe vuote o separate male)
                except IndexError:
                    continue
                except ValueError:
                    continue
            # 3.6 - Close file
            preprocessed_data.close()
            # 3.7 - Extract only 1000 central lines
            with open(preprocessed_path, 'r') as file:
                lines = file.readlines()
            start_index = max(0, len(lines) // 2 - 500)
            end_index = start_index + 1000
            central_lines = lines[start_index:end_index]
            # print check
            # print(m,s,t)
            # 3.8 - Write back only 1000 central lines
            with open(preprocessed_path, 'w') as file:
                file.writelines(central_lines)
print("PRE-PROCESSING PHASE COMPLETED") 