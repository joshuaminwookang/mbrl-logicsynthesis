import csv
with open('test.csv') as csvfile:
    spamreader = csv.reader(csvfile)
    for j, row in enumerate(spamreader):
        if j == 0 : continue
        x = [float(i) for i in row[1:]]
        print(row[0], len(x))
