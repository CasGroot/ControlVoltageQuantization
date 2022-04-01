import os
import torch
from brainspy.utils.io import load_configs
from bspytasks.models.AllInputs import AllInputs
from brainspy.utils.transforms import linear_transform

# path to file where plots should be saved
path = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\inputspace\-1_1\NewAll16'
save_dir = r'C:\Users\CasGr\Documents\github\plots'

# getting the configs for the processor 
configs = load_configs(r"C:\Users\CasGr\Documents\github\brainspy-tasks\configs\defaults\processors\simulation.yaml")

# creating a processor
model = AllInputs(configs)

# Creating 30k uniform random values over the input space and saves them to given file
# x = torch.FloatTensor(30000, 7).uniform_(-1, 1)
# torch.save(x, os.path.join(save_dir, 'random_inputs.pickle'))

# load already created inputs
x = torch.load(os.path.join(path, 'random_inputs.pickle'))

# load quantized input space
x_quant = torch.load(os.path.join(path, 'quantized_input_space.pickle'))

# Runs a forward pass through dnpu surrogate model using generated input space
output_quantized_dict = {}
for i in range(16, 3, -1):
    output = model(x_quant['quantized_input_space_' + str(i)])
    output_quantized_dict['output_' + str(i)] = output
    
# Saves outputs
torch.save(output_quantized_dict, os.path.join(save_dir, 'output_quantized_dict.pickle'))
