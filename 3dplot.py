import os
import torch
from brainspy.utils.io import load_configs
from bspytasks.models.default_ring import DefaultCustomModel
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# path to file where plots should be saved
plot_dir = 'C:/Users/CasGr/Documents/github/plots'

# Creating 30k uniform random values over the input space and saves them to given file
# x = torch.FloatTensor(30000, 2).uniform_(-1, 1)
# torch.save(x, os.path.join(plot_dir, 'random_2d_inputs.pickle'))

# load already created inputs
x = torch.load(os.path.join(plot_dir, 'random_2d_inputs.pickle'))

# load quantized input space
x_quant = torch.load(os.path.join(plot_dir, 'quantized_2d_inputs.pickle'))


# getting the configs for the processor 
configs = load_configs(r"C:\Users\CasGr\Documents\github\brainspy-tasks\configs\defaults\processors\simulation.yaml")

# creating a processor
model = DefaultCustomModel(configs)

model.set_control_voltages(torch.zeros(1, 5))
# Runs a forward pass through dnpu surrogate model using generated input space
output_original = model(x)
output_quantized = model(x_quant['quantized_input_space_6'])

error = (output_original - output_quantized)**2

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter(x[:,0].detach().numpy(), x[:,1].detach().numpy(), error.detach().numpy(), c=error.detach().numpy(), cmap='viridis', linewidth=0.5)
plt.show()