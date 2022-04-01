import os
import torch
from brainspy.utils.io import load_configs
from bspytasks.models.InputSpace import AllInputs
import matplotlib.pyplot as plt
from brainspy.utils.transforms import linear_transform
import numpy as np


def errordistribution(index):
    n = len(index)
    errordist = torch.zeros(n)
    for i in range(0, n):
        errordist[i] = error[int(index[i])]
    return errordist

# path to file where plots should be saved
path = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\inputspace\-1_1'
plot_dir = r'C:\Users\CasGr\Documents\github\plots'

# getting the configs for the processor 
configs = load_configs(r"C:\Users\CasGr\Documents\github\brainspy-tasks\configs\defaults\processors\simulation.yaml")

# creating a processor
model = AllInputs(configs)

# Creating 30k uniform random values over the input space and saves them to given file
# x = torch.FloatTensor(30000, 7).uniform_(-1, 1)
# torch.save(x, os.path.join(path, 'random_inputs.pickle'))

# load already created inputs
x = torch.load(os.path.join(path, 'random_inputs.pickle'))

# load quantized input space
x_quant = torch.load(os.path.join(path, 'quantized_input_space.pickle'))

# Runs a forward pass through dnpu surrogate model using generated input space
output_original = model(x)
output_quantized = model(x_quant['quantized_input_space_6'])

# error = torch.sqrt((output_original - output_quantized)**2)
error = output_original - output_quantized

# plt.scatter(output_original.detach().numpy(), output_quantized.detach().numpy())
# plt.xlabel('original')
# plt.ylabel('quantized')
# plt.show()
index1 = np.array([])
index2 = np.array([])
index3 = np.array([])
index4 = np.array([])
index5 = np.array([])
index6 = np.array([])
index7 = np.array([])
index8 = np.array([])
index9 = np.array([])
index10 = np.array([])
for j in range(0, 7):
    electrode = j
    for i in range(0, x[:,electrode].size()[0]):
        if (-1<=x[i,electrode]<-0.8):
            index1 = np.append(index1, i)
        if (-0.8<=x[i,electrode]<-0.6):
            index2 = np.append(index2, i)
        if (-0.6<=x[i,electrode]<-0.4):
            index3 = np.append(index3, i)
        if (-0.4<=x[i,electrode]<-0.2):
            index4 = np.append(index4, i)
        if (-0.2<=x[i,electrode]<0):
            index5 = np.append(index5, i)
        if (0<=x[i,electrode]<0.2):
            index6 = np.append(index6, i)
        if (0.2<=x[i,electrode]<0.4):
            index7 = np.append(index7, i)
        if (0.4<=x[i,electrode]<0.6):
            index8 = np.append(index8, i)
        if (0.6<=x[i,electrode]<0.8):
            index9 = np.append(index9, i)
        if (0.8<=x[i,electrode]<=1):
            index10 = np.append(index10, i)

    errorlist = [[]]*10
    errorlist[0] = errordistribution(index1).detach().numpy()
    errorlist[1] = errordistribution(index2).detach().numpy()
    errorlist[2] = errordistribution(index3).detach().numpy()
    errorlist[3] = errordistribution(index4).detach().numpy()
    errorlist[4] = errordistribution(index5).detach().numpy()
    errorlist[5] = errordistribution(index6).detach().numpy()
    errorlist[6] = errordistribution(index7).detach().numpy()
    errorlist[7] = errordistribution(index8).detach().numpy()
    errorlist[8] = errordistribution(index9).detach().numpy()
    errorlist[9] = errordistribution(index10).detach().numpy()
    fig = plt.figure()
    # mean = np.array([])
    # std = np.array([])
    # for i in range(0, 10):
    #     mean = np.append(mean, errorlist[i].mean())
    #     std = np.append(std, errorlist[i].std())
    # plt.errorbar(np.linspace(-1,1,10), mean, yerr=std, fmt='o')
    input_ranges = model.get_input_ranges()
    y = linear_transform(-0.9, 0.9, input_ranges[0,0,0], input_ranges[0,0,1], torch.linspace(-0.9,0.9, 10))
    print(np.round(y, decimals=1))
    # plt.xticks(np.linspace(1,10,10), y.detach().numpy(), rotation=-90)
    # plt.violinplot(errorlist)
    # plt.show()

    # fig.savefig(os.path.join(plot_dir, "input" + str(electrode)))
