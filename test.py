import csv

def count_csv_lines(filename):
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader)
    return row_count


materials = ["empty", "acrylic", "aluminium", "brass", "copper", "lignum_vitae", "nylon", "oak", "pine", "pp",
                 "pvc", "rose_wood", "steel"]
tests = ["test1.csv", "test2.csv", "test3.csv", "test4.csv", "test5.csv", "test6.csv", "test7.csv", "test8.csv",
            "test9.csv", "test10.csv"]

tot = 0
for m in materials:
    for t in tests:
        path = f"complete_dataset/{m}/with/{t}"
        tot += count_csv_lines(path)
        path = f"complete_dataset/{m}/without/{t}"
        tot += count_csv_lines(path)
print(tot)