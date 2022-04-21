import os
import torch
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # path to file where plots should be saved
    path = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\inputspace\-1_1\NewModel'
    plot_dir = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\Results\Plots_1_003_nA\MsevsBits'

    # Runs a forward pass through dnpu surrogate model using generated input space
    outputs = torch.load(os.path.join(path, 'output_quantized_dict.pickle'))

    # Define loss function
    loss = torch.nn.MSELoss()

    # Obtain original output
    output_original = outputs['output_16']
    hw_output_original = torch.load(os.path.join(path, 'tmp', 'hw_output16.pickle'))
    hw_rmsearray = np.array([])
    rmsearray = np.array([])
    # Calculate rmse for 16 until 4 bits
    for frac in range(16, 3, -1):
        output_quantized = outputs['output_' + str(frac)]
        rmse = torch.sqrt(loss(output_original, output_quantized))
        rmsearray = np.append(rmsearray, rmse.detach().numpy())
        hw_output_quantized = torch.load(os.path.join(path, 'tmp', f'hw_output{frac}.pickle'))
        hw_rmse = torch.sqrt(loss(hw_output_original, hw_output_quantized))
        hw_rmsearray = np.append(hw_rmsearray, hw_rmse.detach().numpy())


    fig = plt.figure()
    plt.scatter(np.linspace(4, 16, 13), np.sqrt(np.flip(hw_rmsearray)), label='Hardware')
    plt.scatter(np.linspace(4, 16, 13), np.sqrt(np.flip(rmsearray)), label='Surrogate Model')
    # plt.yscale('log')
    plt.legend()
    plt.xlim(17, 3)
    plt.xlabel('number of bits')
    plt.ylabel('RMSE (nA)')
    plt.show()
    fig.savefig(os.path.join(plot_dir, 'hw_RMSEvsBits.png'))
   