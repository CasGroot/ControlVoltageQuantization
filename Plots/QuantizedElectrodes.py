import os
import torch
import numpy as np
from brainspy.utils.io import load_configs
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\inputspace\-1_1\NewModel'
    plot_dir = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\Results\Plots_1_003_nA\Configurations\4bit'

    num_electrodes = 1
    order_electrodes = 'NO'
    save_fig = True
    
    # Obtain outputs
    outputs = torch.load(os.path.join(path, 'AllConfDict4bits.pickle'))
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

        # Plot for 2 quantized electrodes
        plt.scatter(np.linspace(0,20,21), array2)
        plt.xticks(np.linspace(0, 20, 21), list(outputs['two_electrodes'].keys()))
        plt.ylabel('RMSE (nA)')
        plt.xlabel('Quantized Electrodes')
        plt.show()

    if num_electrodes == 3:
        array3 = np.array([])
        for i in range(0, 35):    
            # if '3' in list(outputs[key]['two_electrodes'].keys())[i]:
            array3 = np.append(array3, np.sqrt(loss(output_original, list(outputs['four_electrodes'].values())[i]).detach().numpy()))

        # Plot for 3 quantized electrodes
        plt.scatter(np.linspace(0,34,35), array3)
        plt.xticks(np.linspace(0,34,35), list(outputs['four_electrodes'].keys()))
        plt.xlabel('Quantized Electrodes')
        plt.ylabel('REMSE (nA)')
        plt.show()

    if save_fig == True:
        fig.savefig(os.path.join(plot_dir, f'RMSEvsQuantizedElectrode{num_electrodes}.png'))
