import numpy as np
import torch
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    # path to file location
    # path = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\validation results\DefaultCustomModel\0.3_1_5_solutions_10\infodicts'
    path = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\Results\Solutions\DefaultCustomModel\0.3_1_5_solutions_10\infodicts'
    save_dir = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\Results\Plots_1_003_nA\AccuracyvsBits\DCM\0.3'

    # initialize arrays
    accuracyarray = np.zeros((10, 13))
    rmse = np.zeros((10, 13))
    out_type = 'simulation'
    plot_type = 'accuracy'
    # obtain necessary values
    g=0
    for i in range(0, 10):
        losses = torch.load(os.path.join(path, f"info{i}.pickle"))
        g=0
        if out_type == 'hardware':
            for j in range(16, 3, -1):
                try:
                    accuracyarray[i, g] = losses['accuracy'][g]
                    rmse[i, g] = losses['rmsedict'][f'rmse quant_{j}bits']
                except: 
                    break
                g+=1
        if out_type == 'simulation':
            for j in range(16, 3, -1):
                accuracyarray[i, g] = losses['accuracy'][g]
                rmse[i, g] = losses['rmsedict'][f'rmse output{j}frac{j-2}']
                g += 1

    # Mask missing hardware values
    accuracyarraymas = np.ma.array(accuracyarray)
    accuracyarraymasked = np.ma.masked_where(accuracyarray < 1, accuracyarray)
    z = [[y for y in row if y] for row in accuracyarraymasked.T]

    # Plotting
    
    fig = plt.figure()
    plt.boxplot(accuracyarray)
    plt.xticks(np.linspace(1, 13, 13), np.linspace(16, 4, 13).astype(int))
    plt.xlabel('number of bits')
    if plot_type == 'rmse':
        plt.ylabel('RMSE (nA)') 
        plt.show()  
        fig.savefig(os.path.join(save_dir, 'rmsevsbits0_3gap'))
    if plot_type == 'accuracy':   
        plt.ylabel('accuracy (%)')
        plt.ylim(50,100)
        plt.show()
        fig.savefig(os.path.join(save_dir, 'AccuracyvsBits0_3gap'))
    
    