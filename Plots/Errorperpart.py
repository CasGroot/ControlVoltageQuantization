import os
import torch
import matplotlib.pyplot as plt
from brainspy.utils.transforms import linear_transform
import numpy as np


def errordistribution(index):
    n = len(index)
    errordist = torch.zeros(n)
    for i in range(0, n):
        errordist[i] = error[int(index[i])]
    return errordist

if __name__ == "__main__":
    # path to file where plots should be saved
    path = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\inputspace\-1_1\NewAll16'
    plot_dir = r'C:\Users\CasGr\Documents\github\plots'

    # load already created inputs
    x = torch.load(os.path.join(path, 'random_inputs.pickle'))

    # load quantized input space
    x_quant = torch.load(os.path.join(path, 'quantized_input_space.pickle'))

    # Obtains original and quantized outputs
    outputs = torch.load(os.path.join(path, 'output_quantized_dict_simulation_old_model.pickle'))

    # Select number of bits
    bits = 8
    output_original = outputs['output_16']
    output_quantized = outputs['output_' + str(bits)]

    # Select error
    # error = torch.sqrt(((output_original - output_quantized)**2)/torch.abs(output_original))
    error = output_original - output_quantized

    # Initialize arrays
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
    errorlistdict = {}

    for j in range(0,7):
        if j == 1:
            output_quantized = outputs['output_8']
        elif j == 2:
            output_quantized = outputs['output_6']

        # Find inputs for voltages range of electrode 
        electrode = 0
        for i in range(0, x[:20000,electrode].size()[0]):
            if (-1<=x[i,electrode]<-0.8):
                index1 = np.append(index1, i)
            elif (-0.8<=x[i,electrode]<-0.6):
                index2 = np.append(index2, i)
            elif (-0.6<=x[i,electrode]<-0.4):
                index3 = np.append(index3, i)
            elif (-0.4<=x[i,electrode]<-0.2):
                index4 = np.append(index4, i)
            elif (-0.2<=x[i,electrode]<0):
                index5 = np.append(index5, i)
            elif (0<=x[i,electrode]<0.2):
                index6 = np.append(index6, i)
            elif (0.2<=x[i,electrode]<0.4):
                index7 = np.append(index7, i)
            elif (0.4<=x[i,electrode]<0.6):
                index8 = np.append(index8, i)
            elif (0.6<=x[i,electrode]<0.8):
                index9 = np.append(index9, i)
            elif (0.8<=x[i,electrode]<=1):
                index10 = np.append(index10, i)

        # Find error distribution of inputs in voltage ranges
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
        errorlistdict['electrode' + str(j)] = errorlist
        print(j)
        # torch.save(errorlistdict, os.path.join(plot_dir, 'errorlist.pickle'))
        
        # Apply linear transform to input range
        input_ranges = torch.load(os.path.join(path, 'quick_input_ranges.pickle'))
        y = linear_transform(-1, 1, input_ranges[0,j,0], input_ranges[0,j,1], torch.linspace(-1,1, 11))
        xaxis = []
        
        for i in range(0, y.size()[0]-1):
            xaxis.append([np.round(y[i].detach().numpy(), decimals=2), np.round(y[i+1].detach().numpy(), decimals=2)])
        
        # Plot
        plt.violinplot(errorlist)
        plt.xticks(np.linspace(1,10,10), xaxis, rotation=-90)
        # plt.ylim(-2,2)
        plt.title(str(bits) + 'bits')
        plt.show()
        fig.savefig(os.path.join(plot_dir, "input" + str(electrode)))
