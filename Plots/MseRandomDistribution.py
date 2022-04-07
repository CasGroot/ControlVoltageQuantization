import os
import torch
import matplotlib.pyplot as plt
import numpy as np
# path to file where plots should be saved
path = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\inputspace\-1_1\NewAll16'
plot_dir = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\Results\Plots\MsevsBits'

# Runs a forward pass through dnpu surrogate model using generated input space
outputs = torch.load(os.path.join(path, 'output_quantized_dict_simulation_old_model.pickle'))

# Define loss function
loss = torch.nn.MSELoss()

# Obtain original output
output_original = outputs['output_16']
msearray = np.array([])

# Calculate rmse for 16 until 4 bits
for frac in range(16, 3, -1):
    output_quantized = outputs['output_' + str(frac)]
    mse = torch.sqrt(loss(output_original, output_quantized))
    msearray = np.append(msearray, mse.detach().numpy())

# Plot rmse vs number of bits
fig = plt.figure()
plt.scatter(np.linspace(4, 16, 13), np.sqrt(np.flip(msearray)))
plt.xlim(17, 3)
plt.xlabel('number of bits')
plt.ylabel('RMSE [nA]')
plt.show()
fig.savefig(os.path.join(plot_dir, 'RMSEvsBits.png'))
    