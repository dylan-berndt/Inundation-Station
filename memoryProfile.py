import numpy as np

memory = np.load("memory.npy")

for line in np.diff(memory, axis=1):
    print(" | ".join([str(num) for num in line]))

for line in np.diff(memory, axis=0):
    print(" | ".join([str(num) for num in line]))
