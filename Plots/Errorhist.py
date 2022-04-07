import os
import torch
import matplotlib.pyplot as plt
import scipy.stats as ss
import scipy.optimize as so

# path to file where plots should be saved
path = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\inputspace\-1_1\NewAll16'
plot_dir = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\Results\Plots\OriginalvsQuantized'

# Runs a forward pass through dnpu surrogate model using generated input space
outputs = torch.load(os.path.join(path, 'output_quantized_dict_simulation_old_model.pickle'))

output_original = outputs['output_16']

# Plot Error Histogram
for frac in range(16, 3, -1):
    output_quantized = outputs['output_' + str(frac)]

    error = output_original - output_quantized

    fig = plt.figure()

    # Plots histogram
    h, bins, patches = plt.hist(error.detach().numpy(), bins=100, density=1)
    for item in patches:
        item.set_height(item.get_height()/sum(h))
    plt.ylim(0, 0.3)

    # Plots original vs quantized output
    # plt.scatter(output_original.detach().numpy(), output_quantized.detach().numpy(), s=5)
    # plt.xlabel('output 16 bits')
    # plt.ylabel('output ' + str(frac) + ' bits')

    plt.show()

    # save figure
    # fig.savefig(os.path.join(plot_dir, "OriginalvsQuantized" + str(frac) + ".png"))
