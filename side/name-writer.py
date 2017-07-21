import csv
import random

names = []

with open("names.csv", "rt") as f:
	reader = csv.reader(f)
	for row in reader:
		names.append(row[1])

random.shuffle(names)

final_names = []

with open("agent-names.csv", "wt") as f:
	writer = csv.writer(f)
	for name in enumerate(names):
		if name[0] % 25 == 0:
			writer.writerow([name[1]])
			final_names.append(name[1])

print(len(final_names))