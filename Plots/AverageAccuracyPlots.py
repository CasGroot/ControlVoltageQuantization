import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# path to file location
path = r'C:\Users\CasGr\Documents\github\brainspy-tasks\tmp\ring\searcher_0.3gap_2022_04_02_164116\reproducibility'
save_dir = r''

# initialize arrays
accuracyarray = np.zeros((2, 13))
rmse = np.zeros((2, 13))

# obtain necessary values
g=0
for i in range(0, 2):
    losses = torch.load(os.path.join(path, "info" + str(i) + ".pickle"))
    g=0
    for j in range(16, 3, -1):
        accuracyarray[i, g] = losses['accuracy'][g]
        rmse[i, g] = losses['rmsedict']['rmse output' + str(j)]
        g+=1

# plotting
plt.boxplot(rmse)
plt.xticks(np.linspace(1, 13, 13), np.linspace(16, 4, 13).astype(int))
plt.xlabel('number of bits')
plt.ylabel('accuracy [%]')
plt.show()