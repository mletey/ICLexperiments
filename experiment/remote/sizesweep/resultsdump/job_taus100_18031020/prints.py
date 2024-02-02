import matplotlib.pyplot as plt
import numpy as np

# Loop over the filtered files
vals = []
for i in range(100):
    filepath = f'./error-P{i}.txt'
    with open(filepath, 'r') as file:
        file_contents = file.read()
    vals.append(float(file_contents))

plt.plot(range(1,101), vals,label="test error")
plt.xlabel("tau")
plt.title("1 Layers, 100 hidden dim")
plt.savefig("errorplot-1-100.png")