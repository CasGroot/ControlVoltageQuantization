import os
import torch
from brainspy.utils.io import load_configs
from bspytasks.models.default_ring_cas import DefaultCustomModel
from bspytasks.models.Architecture21 import Architecture21
from brainspy.algorithms.modules.performance.accuracy import get_accuracy
from brainspy.utils.manager import get_criterion
from bspytasks.ring.tasks.classifier import plot_perceptron
import matplotlib.pyplot as plt
import yaml
from brainspy.utils.pytorch import TorchUtils


def plot_output(original, quantized, key):
    # plotting output of original and quantized model in one plot
    plt.figure()
    plt.plot(original.detach().numpy(), c='blue', label='original')
    plt.plot(quantized.detach().numpy(), c='r', label=key)
    plt.title('output (nA)')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, key + ".jpg"))


def subplot_output(original, quantized, key):
    # plotting output of original and quantized model in subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('output (nA)')
    ax1.plot(original.detach().numpy())
    ax1.set_title('original')
    ax2.plot(quantized.detach().numpy())
    ax2.set_title('quantized : {}'.format(key))
    fig.savefig(os.path.join(plot_dir, "subplot_" + key + ".jpg"))


def absmse_append(original, quantized, key):
    # append dictionary of mse without offset
    absdiff = original - quantized
    if (torch.sum(absdiff**2)/absdiff.size()[0] is not None):
        absmsedict['absolute mse ' + key] = torch.sum(absdiff**2)/absdiff.size()[0]
    else:
        print('was none')


def relmse_append(original, quantized, key):
    # append dictionary of mse with offset
    meandiff = torch.mean(quantized) - torch.mean(original)
    shiftedquantized = quantized - meandiff
    reldiff = original - shiftedquantized
    if (torch.sum(reldiff**2)/reldiff.size()[0] is not None):
        relmsedict['relative mse ' + key] = torch.sum(reldiff**2)/reldiff.size()[0]


def rmse_append(original, quantized, key):
    # append dictionary of rmse
    absdiff = original - quantized
    if (torch.sqrt(torch.sum(absdiff**2)/absdiff.size()[0]) is not None):
        rmsedict['rmse ' + key] = torch.sqrt(torch.sum(absdiff**2)/absdiff.size()[0])
    else:
        print('was none')


def mean_append(mean1, mean2, std, i):
    # append dictionary of mean and std
    mean1dict[i] = mean1
    mean2dict[i] = mean2
    stddict[i] = std


def plot_accuracyvsbits(accuracyarray):
    # plotting accuracy vs bits
    fig = plt.figure()
    plt.scatter(torch.linspace(16, 4, 13), accuracyarray)
    plt.xlabel('number of bits')
    plt.ylabel('accuracy')
    fig.savefig(os.path.join(plot_dir, "accuracy_vs_bits.jpg"))


def save_dicts():
    # save all dictionaries
    dicts = {
            'absmsedict': absmsedict,
            'relmsedict': relmsedict,
            'rmsedict': rmsedict,
            'mean1dict': mean1dict,
            'mean2dict': mean2dict,
            'stddict': stddict,
            'accuracy': accuracyarray,
            }
    torch.save(dicts, os.path.join(plot_dir, 'info.pickle'))


if __name__ == "__main__":
    dir = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\Results\Solutions\DefaultCustomModel\0.3_1_5_solutions_10'
    for searchfile in os.listdir(dir):
        if searchfile != 'infodicts':       
            # path to reproducibility file
            path = os.path.join(dir, searchfile, 'reproducibility')
            # path to file where plots should be saved
            plot_dir = os.path.join(path, 'plots')

            # loading in necessary files
            configs = load_configs(r'C:\Users\CasGr\Documents\github\brainspy-tasks\configs\ring.yaml')
            results = torch.load(os.path.join(path, "results.pickle"), map_location=TorchUtils.get_device())
            training_data = torch.load(os.path.join(path, 'training_data.pickle'), map_location=TorchUtils.get_device())

            model = DefaultCustomModel(configs['processor'])
            model.load_state_dict(training_data['model_state_dict'])
            # train_output = torch.load(os.path.join(path, 'ring_train_output_dict.pickle'))
            test_output = torch.load(os.path.join(path, 'ring_test_output_dict.pickle'), map_location=TorchUtils.get_device())

            # initialize dictionaries
            absmsedict = {}
            relmsedict = {}
            rmsedict = {}
            mean1dict = {}
            mean2dict = {}
            stddict = {}
            accuracyarray = torch.Tensor(0)

            i = 16

            for key in test_output:
                
                # finding the accuracy of the model on test data
                for param in model.parameters():
                    param.requires_grad = False 
                accuracy_dict_test = get_accuracy(model(results['train_results']['inputs']), results['train_results']['targets'], configs['accuracy'])#, node = results['train_results']['accuracy']['node'])

                # plotting perceptron 
                plot_perceptron(accuracy_dict_test, save_dir=plot_dir, name="quant_" + key, show_plots=True)

            #     # plotting outpout
            #     subplot_output(results['test_results']['best_output'], test_output[key], key)
            #     plot_output(results['test_results']['best_output'], test_output[key], key)

            #     # calculating different losses for output of the model
            #     absmse_append(results['test_results']['best_output'], test_output[key], key)
            #     relmse_append(results['test_results']['best_output'], test_output[key], key)
            #     rmse_append(results['test_results']['best_output'], test_output[key], key)

            #     # adding accuracy values to array
            #     accuracyarray = torch.cat((accuracyarray, accuracy_dict_test['accuracy_value'].view(1)))
            #     i -= 1
            
            # save_dicts()
