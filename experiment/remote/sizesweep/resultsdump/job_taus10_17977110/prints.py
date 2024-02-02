import matplotlib.pyplot as plt
import numpy as np

# Loop over the filtered files
vals = []
for i in range(60):
    filepath = f'./error-P{i}.txt'
    with open(filepath, 'r') as file:
        file_contents = file.read()
    vals.append(float(file_contents))

plt.plot(np.linspace(0.1,10,60),vals,label="test error")
plt.xlabel("tau")
plt.title("2 Layers, 10 hidden dim")
plt.savefig("errorplot-2-10.png")