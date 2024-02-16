import matplotlib.pyplot as plt
import numpy as np
import sys

myjob = sys.argv[1]

# Loop over the filtered files
vals = []
for i in range(60):
    filepath = f'./{myjob}/errors/error-{i}.txt'
    with open(filepath, 'r') as file:
        file_contents = file.read()
    vals.append(float(file_contents))

plt.plot(range(1,61), vals,label="test error")
plt.xlabel("tau")
plt.title("2 Layers, 100 hidden dim, d = 7, k = P")
plt.savefig(f'./{myjob}/errorplot-2-100.png')