import os
import torch
import numpy as np
from brainspy.utils.io import load_configs
import matplotlib.pyplot as plt

path = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\inputspace\-1_1\NewModel'
plot_dir = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\Results\Plots_1_003_nA\Configurations'

# Obtain outputs
outputs = torch.load(os.path.join(path, 'AllConfDict8bits.pickle'))
output_original = outputs['no_electrodes']

# Initialize necessary variables
loss = torch.nn.MSELoss()
array = np.array([])

# Obtain error for every configuration
for i in range(0, 7):
    array = np.append(array, np.sqrt(loss(output_original, list(outputs['one_electrode'].values())[i]).detach().numpy()))

# Used to order for 1 quantized electrode
order = np.array([6,4,2,0,1,3,5])
orderedarray = np.ones(7)

# Order configurations
for i in range(0, 7):
    orderedarray[order[i]] = array[i]

# Plot error
fig = plt.figure()
# Plot for 1 quantized electrode
# plt.scatter(np.linspace(0,6,7), orderedarray)
# plt.ylabel('RMSE (nA)')
# plt.xlabel('Quantized Electrodes')
# plt.show()
# fig.savefig(os.path.join(plot_dir, 'RMSEvsQuantizedElectrodeordered.png'))

array2 = np.array([])
for i in range(0, 21):    
    array2 = np.append(array2, np.sqrt(loss(output_original, list(outputs['two_electrodes'].values())[i]).detach().numpy()))

# Plot for 2 quantized electrodes
# plt.scatter(np.linspace(0,20,21), array2)
# plt.xticks(np.linspace(0, 20, 21), list(outputs['two_electrodes'].keys()))
# plt.ylabel('RMSE (nA)')
# plt.xlabel('Quantized Electrodes')
# plt.show()

array3 = np.array([])
for i in range(0, 35):    
    # if '3' in list(outputs[key]['two_electrodes'].keys())[i]:
    array3 = np.append(array3, np.sqrt(loss(output_original, list(outputs['three_electrodes'].values())[i]).detach().numpy()))

# Plot for 3 quantized electrodes
plt.scatter(np.linspace(0,34,35), array3)
plt.xticks(np.linspace(0,34,35), list(outputs['three_electrodes'].keys()))
plt.xlabel('Quantized Electrodes')
plt.ylabel('REMSE (nA)')
plt.show()

fig.savefig(os.path.join(plot_dir, 'RMSEvsQuantizedElectrode.png'))
