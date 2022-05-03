import os
import torch
import matplotlib.pyplot as plt
import scipy.stats as ss
import scipy.optimize as so

# path to file where plots should be saved
path = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\inputspace\-1_1\NewModel'
plot_dir = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\Results\Plots\OriginalvsQuantized'

# Loads quantized output from pickle file
outputs = torch.load(os.path.join(path, 'output_quantized_dict.pickle'))
output_original = outputs['output_16']

# Load hw_outputs
hw_output_original = torch.load(os.path.join(path, 'tmp', 'hw_output16.pickle'))

# Plot Error Histogram
for bits in range(8, 7, -1):
    output_quantized = outputs['output_' + str(bits)]
    hw_output_quantized = torch.load(os.path.join(path, 'tmp', f'hw_output{bits}.pickle'))
    error = output_original - output_quantized
    hw_error = hw_output_original - hw_output_quantized
    fig = plt.figure()

    # Plots histogram
    h, bins, patches = plt.hist(error.detach().numpy(), bins=100)
    # for item in patches:
    #     item.set_height(item.get_height()/60)
    # plt.ylim(0, 10000)

    # Plots original vs quantized output
    # plt.scatter(output_original.detach().numpy(), output_quantized.detach().numpy(), s=5)
    # plt.scatter(hw_output_original.detach().numpy(), hw_output_quantized.detach().numpy(), s=5)
    # plt.xlabel('output 16 bits')
    # plt.ylabel('output ' + str(bits) + ' bits')

    plt.show()

    # save figure
    # fig.savefig(os.path.join(plot_dir, "OriginalvsQuantized" + str(bits) + ".png"))
    # fig.savefig(os.path.join(plot_dir, f'hw_OriginalvsQuantized{bits}.png')
