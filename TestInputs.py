import os
import torch
from brainspy.utils.io import load_configs
from bspytasks.models.InputSpace import AllInputs
from brainspy.processors.processor import Processor
import matplotlib.pyplot as plt
from brainspy.utils.transforms import linear_transform
# path to file where plots should be saved
path = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\inputspace\-1_1'
plot_dir = r'C:\Users\CasGr\Documents\github\plots'

# Creating 30k uniform random values over the input space and saves them to given file
# x = torch.FloatTensor(30000, 7).uniform_(-1, 1)
# torch.save(x, os.path.join(path, 'random_inputs.pickle'))

# load already created inputs
x = torch.load(os.path.join(path, 'random_inputs.pickle'))

# load quantized input space
x_quant = torch.load(os.path.join(path, 'quantized_input_space.pickle'))

# getting the configs for the processor 
configs = load_configs(r"C:\Users\CasGr\Documents\github\brainspy-tasks\configs\defaults\processors\simulation.yaml")

# creating a processor
model = AllInputs(configs)

# Runs a forward pass through dnpu surrogate model using generated input space
output_original = model(x)
output_quantized = model(x_quant['quantized_input_space_8'])

error = torch.sqrt((output_original - output_quantized)**2)/torch.abs(output_original)

size = x.size()[0]
percent = 0.1 # percentage of error to be taken
n = int(size*percent)

index = torch.argsort(error, dim=0)
input1 = torch.zeros(n)
input2 = torch.zeros(n)
input3 = torch.zeros(n)
input4 = torch.zeros(n)
input5 = torch.zeros(n)
input6 = torch.zeros(n)
input7 = torch.zeros(n)

for i in range(size-n, size):
    input1[i-size+n] = x[index[i], 0]
    input2[i-size+n] = x[index[i], 1]
    input3[i-size+n] = x[index[i], 2]
    input4[i-size+n] = x[index[i], 3]
    input5[i-size+n] = x[index[i], 4]
    input6[i-size+n] = x[index[i], 5]
    input7[i-size+n] = x[index[i], 6]

input_ranges = model.get_input_ranges()

transformed_input1 = linear_transform(input1.min(), input1.max(), input_ranges[0,0,0], input_ranges[0,0,1], input1) 
transformed_input2 = linear_transform(input2.min(), input2.max(), input_ranges[0,1,0], input_ranges[0,1,1], input2) 
transformed_input3 = linear_transform(input3.min(), input3.max(), input_ranges[0,2,0], input_ranges[0,2,1], input3) 
transformed_input4 = linear_transform(input4.min(), input4.max(), input_ranges[0,3,0], input_ranges[0,3,1], input4) 
transformed_input5 = linear_transform(input5.min(), input5.max(), input_ranges[0,4,0], input_ranges[0,4,1], input5) 
transformed_input6 = linear_transform(input6.min(), input6.max(), input_ranges[0,5,0], input_ranges[0,5,1], input6) 
transformed_input7 = linear_transform(input7.min(), input7.max(), input_ranges[0,6,0], input_ranges[0,6,1], input7) 

fig, axs= plt.subplots(2, 4)
axs[0,0].hist(transformed_input1.detach().numpy(), bins=20)
axs[0,0].set_xlabel('input1')
axs[0,0].set_ylabel('count')

axs[0,1].hist(transformed_input2.detach().numpy(), bins=20)
axs[0,1].set_xlabel('input2')
axs[0,1].set_ylabel('count')

axs[0,2].hist(transformed_input3.detach().numpy(), bins=20)
axs[0,2].set_xlabel('input3')
axs[0,2].set_ylabel('count')

axs[0,3].hist(transformed_input4.detach().numpy(), bins=20)
axs[0,3].set_xlabel('input4')
axs[0,3].set_ylabel('count')

axs[1,0].hist(transformed_input5.detach().numpy(), bins=20)
axs[1,0].set_xlabel('input5')
axs[1,0].set_ylabel('count')

axs[1,1].hist(transformed_input6.detach().numpy(), bins=20)
axs[1,1].set_xlabel('input6')
axs[1,1].set_ylabel('count')

axs[1,2].hist(transformed_input7.detach().numpy(), bins=20)
axs[1,2].set_xlabel('input7')
axs[1,2].set_ylabel('count')

plt.show()    
fig.savefig(os.path.join(plot_dir, "inputHistogram.jpg"))   
