import os
import torch
from brainspy.utils.io import load_configs
from bspytasks.models.default_ring import DefaultCustomModel
from bspytasks.models.Architecture21 import Architecture21
from brainspy.algorithms.modules.performance.accuracy import get_accuracy
from brainspy.utils.manager import get_criterion
from bspytasks.ring.tasks.classifier import plot_perceptron
import matplotlib.pyplot as plt
import yaml

    
def plot_output(original, quantized, key, show = False):
    # plotting output of original and quantized model
    plt.figure()
    plt.plot(original, c='blue', label='original')
    plt.plot(quantized, c='r', label=key)
    plt.title('output (nA)')
    plt.legend()
    if show:
        plt.show()
    plt.savefig(os.path.join(plot_dir, "output" + key + ".jpg"))

def subplot_output(original, quantized, key, show = False):
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle('output (nA)')
    ax1.plot(original)
    ax1.set_title('original')
    ax2.plot(quantized)
    ax2.set_title('quantized : {}'.format(key))
    fig.savefig(os.path.join(plot_dir, "subplot_output" + key + ".jpg"))

def absmse_append(original, quantized, key):
    absdiff = original - quantized
    if (torch.sum(absdiff**2)/absdiff.size()[0] is not None):
        absmsedict['absolute mse ' + key] = torch.sum(absdiff**2)/absdiff.size()[0]
    else:
        print('was none')
    

def relmse_append(original, quantized, key):
    meandiff = torch.mean(quantized) - torch.mean(original)
    shiftedquantized = quantized - meandiff
    reldiff = original - shiftedquantized
    if (torch.sum(reldiff**2)/reldiff.size()[0] is not None):
        relmsedict['relative mse ' + key] = torch.sum(reldiff**2)/reldiff.size()[0]

def rmse_append(original, quantized, key):
    absdiff = original - quantized
    if (torch.sqrt(torch.sum(absdiff**2)/absdiff.size()[0]) is not None):
        rmsedict['rmse ' + key] = torch.sqrt(torch.sum(absdiff**2)/absdiff.size()[0])
    else:
        print('was none')

def save_dicts():
    with open(os.path.join(plot_dir, 'losses.yaml'), 'w') as file:
        yaml.dump(str(absmsedict), file)
        yaml.dump(str(relmsedict), file)
        yaml.dump(str(rmsedict), file)
        yaml.dump(str(meandict), file)
        yaml.dump(str(stddict), file)
    dicts = {'absmse': absmsedict, 
            'relmsedict': relmsedict,
            'rmsedict': rmsedict,
            'meandict': meandict, 
            'stddict': stddict,
            }
    torch.save(dicts, 'info.pickle')

# path to reproducibility file
path= r'C:\Users\CasGr\Documents\github\brainspy-tasks\tmp\ring\searcher_0.5gap_2022_02_28_152609\reproducibility'
# path to file where plots should be saved
plot_dir= 'C:/Users/CasGr/Documents/github/plots'

# loading in necessary files
loss_fn = get_criterion("fisher")
configs = load_configs(os.path.join(path, "configs.yaml"))
results = torch.load(os.path.join(path, "results.pickle"))

# model_state_dict = torch.load(os.path.join(path, "model.pt"))
training_data = torch.load(os.path.join(path, "training_data.pickle"))
quantized_control_voltages = torch.load(os.path.join(path, "quantized_control_voltages.pickle"))
# quantized_control_voltages2 = {}
# quantized_control_voltages2['original'] = quantized_control_voltages['original']

# creating a new model instance and setting quantized control voltages
new_model_instance = DefaultCustomModel(configs['processor'])
model = 'DCM'
# new_model_instance = Architecture21(configs['processor'])
# model = 'Arch21'
new_model_instance.load_state_dict(training_data['model_state_dict'])

plot_perceptron(results['test_results']['accuracy'], save_dir=plot_dir, name="test")

# initialize dictionaries
absmsedict = {}
relmsedict = {}
rmsedict = {}
meandict = {}
stddict = {}

for key in quantized_control_voltages:
    new_model_instance.set_control_voltages(quantized_control_voltages[key])
    # finding predictions using quantized control voltages
    for param in new_model_instance.parameters():
        param.requires_grad = False 
    if (model == 'Arch21'):
        prediction_train, output = new_model_instance(results['train_results']['inputs'])
        prediction_test, output = new_model_instance(results['test_results']['inputs'])
        mean = torch.mean(output['c_dnpu_output'])
        std = torch.std(output['c_dnpu_output'])
        print('{} mean: {}'.format(key, mean))
        print('{} std: {}'.format(key, std))
        meandict['mean ' + key] = mean
        stddict['std ' + key] = std
    elif (model == 'DCM'):
        prediction_train = new_model_instance(results['train_results']['inputs'])
        prediction_test = new_model_instance(results['test_results']['inputs'])
    
    # training a perceptron to find a suitable threshold
    accuracy_dict_training = get_accuracy(prediction_train, results['train_results']['targets'], configs['accuracy'], node=None)
    
    # finding the accuracy of the model on test data
    accuracy_dict_test = get_accuracy(prediction_test, results['test_results']['targets'], configs['accuracy'], node = accuracy_dict_training['node'])

    # plotting perceptron 
    plot_perceptron(accuracy_dict_test, save_dir=plot_dir, name="quant_" + key)

    subplot_output(results['test_results']['best_output'], prediction_test, key, show=False)
    plot_output(results['test_results']['best_output'], prediction_test, key, show=False)

    # calculating different losses for output of the model
    # fisherloss = loss_fn(prediction_test, results['test_results']['targets'])
    absmse_append(results['test_results']['best_output'], prediction_test, key)
    relmse_append(results['test_results']['best_output'], prediction_test, key)
    rmse_append(results['test_results']['best_output'], prediction_test, key)

    save_dicts()


    

    