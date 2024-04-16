import matplotlib.pyplot as plt
import re
import numpy

def round_accuracies_in_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    def round_accuracy(match):
        numbers = match.group(1).split(', ')
        rounded_numbers = [str(round(float(num), 2)) for num in numbers]
        return 'accuracy=[' + ', '.join(rounded_numbers) + ']'

    updated_content = re.sub(r'accuracy=\[(.*?)\]', round_accuracy, content)

    with open(file_path, 'w') as file:
        file.write(updated_content)

def extract_accuracy_from_file(file_path):
    # This function reads the file and extracts accuracy data
    accuracy_lists = []
    epochs = []

    # Open the file and read through each line
    with open(file_path, 'r') as file:
        content = file.read()

        # Find all occurrences of accuracy lists in the file
        matches = re.findall(r'epochs=(\d+)\s+accuracy=\[(.*?)\]', content)
        
        # For each match, convert the string of numbers into a list of floats
        for epoch, match in matches:
            accuracy_list = list(map(float, match.split(',')))
            accuracy_lists.append(accuracy_list)
            epochs.append(int(epoch))

    return accuracy_lists, epochs


def plot_results():
    path = 'results.txt'    
    
    acc, ep = extract_accuracy_from_file(path)
    for idx, a in enumerate(acc):
        plt.plot(numpy.arange(start=1, stop=ep[idx]+1), a)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
    plt.savefig('plots/results.png')
    
plot_results()