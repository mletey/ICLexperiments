import matplotlib.pyplot as plt
import numpy as np

# Loop over the filtered files
vals = []
for i in range(100):
    filepath = f'./error-P{i}.txt'
    with open(filepath, 'r') as file:
        file_contents = file.read()
    vals.append(float(file_contents))

print("max 80",np.where(vals == np.max(vals[80:-1])))
print("min 80",np.where(vals == np.min(vals[80:-1])))
print("max 20",np.where(vals == np.max(vals[20:40])))
print("min 20",np.where(vals == np.min(vals[20:40])))

plt.plot(range(1,101), vals,label="test error")
plt.xlabel("tau")
plt.title("2 Layers, 100 hidden dim")
# plt.savefig("errorplot-2-100.png")