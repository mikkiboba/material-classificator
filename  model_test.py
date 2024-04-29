from net import Cnn2
import torch

def extract_values():
    path = 'complete_dataset/rose_wood/with/test1.csv'
    size = 64
    values = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for i in range(size):
            val = lines[i].strip().split(',')[25]
            val = val[1:len(val)-1].strip().split(" ")
            val = [float(v) for v in val]
            values.append(val)
    return values



model = Cnn2()

model.load_state_dict(torch.load('model.pth'))

model.eval()

'''data = "110 96 6 0 0 0 0 0 0 0 0 0 -28 -7 -29 -5 -32 -4 -35 -2 -35 -3 -34 -4 -35 -4 -34 -6 -34 -6 -32 -7 -32 -6 -31 -7 -27 -6 -23 -5 -19 -5 -17 -3 -12 -2 -9 1 -6 3 -5 6 -4 9 -4 13 -4 13 -5 15 -6 15 -7 13 0 0 -4 9 -2 8 -1 7 1 7 2 6 4 7 5 8 5 9 6 10 5 11 4 13 3 13 2 13 1 14 -1 14 -2 14 -2 15 -3 14 -4 14 -4 14 -4 15 -4 16 -5 17 -4 17 -5 17 -4 16 0 0 0 0 0 0 0 0 0 0"
data = data.split(" ")
data = [int(v) for v in data]
data = torch.tensor(data)'''

data = extract_values()
data = torch.tensor(data)


with torch.no_grad():
    pred = model(data)
    correct = pred.argmax(1)

materials = ["empty", "acrylic", "aluminium", "brass", "copper", "lignum_vitae", "nylon", "oak", "pine", "pp",
                 "pvc", "rose_wood", "steel"]

print(correct)