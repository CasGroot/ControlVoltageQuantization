import os
from matplotlib.transforms import Bbox
import torch
import numpy as np
from brainspy.utils.io import load_configs
import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_electrodes = 3
    order_electrodes = 'NO'
    save_fig = True
    bits = 6
    path = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\inputspace\-1_1\NewModel'
    plot_dir = rf'C:\Users\CasGr\Documents\uni\BachelorOpdracht\Results\Plots_1_003_nA\Configurations\{bits}bit'

    
    
    # Obtain outputs
    outputs = torch.load(os.path.join(path, f'AllConfDict{bits}bits.pickle'))
    output_original = outputs['no_electrodes']

    # Initialize necessary variables
    loss = torch.nn.MSELoss()
    array = np.array([])
    if num_electrodes == 1:
        # Obtain error for every configuration
        for i in range(0, 7):
            array = np.append(array, np.sqrt(loss(output_original, list(outputs['one_electrode'].values())[i]).detach().numpy()))

        if order_electrodes == 'YES':
            # Used to order for 1 quantized electrode
            order = np.array([6,4,2,0,1,3,5])
            orderedarray = np.ones(7)

            # Order configurations
            for i in range(0, 7):
                orderedarray[order[i]] = array[i]

        # Plot error
        fig = plt.figure()
        plt.scatter(np.linspace(0,6,7), array)
        plt.ylabel('RMSE (nA)')
        plt.xlabel('Quantized Electrodes')
        plt.show()

    if num_electrodes == 2 :
        array2 = np.array([])
        for i in range(0, 21):    
            array2 = np.append(array2, np.sqrt(loss(output_original, list(outputs['two_electrodes'].values())[i]).detach().numpy()))
        xlabel_list = [[]]*21
        g=0
        for a in range(0, 7):
            for b in range(a+1, 7):
                xlabel_list[g] = [a,b]
                g+=1
        # Plot for 2 quantized electrodes
        fig = plt.figure()
        plt.scatter(np.linspace(0,20,21), array2)
        plt.xticks(np.linspace(0, 20, 21), xlabel_list, rotation=-90)
        plt.ylabel('RMSE (nA)')
        plt.xlabel('Quantized Electrodes')
        plt.show()

    if num_electrodes == 3:
        array3 = np.array([])
        for i in range(0, 35):    
            # if '3' in list(outputs[key]['two_electrodes'].keys())[i]:
            array3 = np.append(array3, np.sqrt(loss(output_original, list(outputs['four_electrodes'].values())[i]).detach().numpy()))
        xlabel_list = [[]]*35
        g=0
        for a in range(0, 7):
            for b in range(a+1, 7):
                for c in range(b+1, 7):
                    xlabel_list[g] = [a,b,c]
                    g+=1

        # Plot for 3 quantized electrodes
        fig = plt.figure(figsize=(10,10))
        plt.scatter(np.linspace(0,34,35), array3)
        plt.xticks(np.linspace(0,34,35), list(outputs['three_electrodes'].keys()))
        # plt.xticks(np.linspace(0,34,35), xlabel_list, rotation=-90)
        plt.xlabel('Quantized Electrodes')
        plt.ylabel('REMSE (nA)')
        plt.show()

    if save_fig == True:
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'RMSEvsQuantizedElectrode{num_electrodes}.png'))
