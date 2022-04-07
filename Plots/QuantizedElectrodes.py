import os
import torch
import numpy as np
from brainspy.utils.io import load_configs
import matplotlib.pyplot as plt

path = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\inputspace\-1_1'
plot_dir = r'C:\Users\CasGr\Documents\github\plots'

# Obtain outputs
outputs = torch.load(os.path.join(plot_dir, 'AllConfDict.pickle'))
output_original = outputs['quantized_input_space_8']['no_electrodes']
# Initialize necessary variables
loss = torch.nn.MSELoss()
key = 'quantized_input_space_8'
array = np.array([])

# Obtain error for every configuration
for i in range(0, 7):
    array = np.append(array, np.sqrt(loss(output_original, list(outputs[key]['one_electrode'].values())[i]).detach().numpy()))

# Plot error
plt.scatter(np.linspace(0,6,7), array)
plt.xticks(np.linspace(0, 6, 7), np.array([0,4,5,6,2,3,1]))
plt.show()
