import os
import torch
from brainspy.utils.io import load_configs
from bspytasks.models.default_ring import DefaultCustomModel
from bspytasks.models.Architecture21 import Architecture21
from bspytasks.models.InputSpace import AllInputs
from brainspy.processors.processor import Processor
import matplotlib.pyplot as plt
import time
import numpy as np

# Save seconds since epoch
t1 = time.time()

# path to file where plots should be saved
plot_dir = 'C:/Users/CasGr/Documents/github/plots'

# Creating 30k uniform random values over the input space and saves them to given file
x = torch.FloatTensor(30000, 7).uniform_(-1, 1)
torch.save(x, os.path.join(plot_dir, 'random_inputs.pickle'))

# load already created inputs
# x = torch.load(os.path.join(plot_dir, 'input_space.pickle'))
# load quantized input space
x_quant = torch.load(os.path.join(plot_dir, 'quantized_input_space.pickle'))


# getting the configs for the processor
configs = load_configs(r"C:\Users\CasGr\Documents\github\brainspy-tasks\configs\defaults\processors\simulation.yaml")

# creating a processor
training_data = torch.load('C:/Users/CasGr/Documents/Data/training_data_quick.pt', map_location=torch.device('cpu'))
processor = Processor(
    configs,
    info=training_data['info'],
    model_state_dict=training_data['model_state_dict'])


# Runs a forward pass through dnpu surrogate model using generated input space
output_original = processor(x)
output_quantized = processor(x_quant['quantized_input_space_6'])

# Calculates mse of the output of original vs quantized
diff = output_original - output_quantized
mse = torch.sum(diff**2)/diff.size()[0]
print(mse)

# Saves time after running and prints 
t2 = time.time()
t = t2 - t1
print(t2 - t1)
print('totaltime: {}'.format(t * 2**7 * 5))
# plt.plot(output_original.detach().numpy())
# plt.show()
                             

