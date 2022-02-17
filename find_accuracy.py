import os
import torch
from brainspy.utils.io import load_configs
from bspytasks.models.default_ring import DefaultCustomModel
from brainspy.algorithms.modules.performance.accuracy import get_accuracy
from brainspy.utils.manager import get_criterion
from bspytasks.ring.tasks.classifier import plot_perceptron
import matplotlib.pyplot as plt

#path to reproducibility file
path= 'C:/Users/CasGr/Documents/github/brainspy-tasks/tmp/ring/searcher_0.5gap_2022_02_14_200312/reproducibility/'
#path to file where plots should be saved
plot_dir= 'C:/Users/CasGr/Documents/github/plots'

#loading in necessary files
loss_fn = get_criterion("fisher")
configs = load_configs(os.path.join(path, "configs.yaml"))
results = torch.load(os.path.join(path, "results.pickle"))
model_state_dict = torch.load(os.path.join(path, "model.pt"))
training_data = torch.load(os.path.join(path, "training_data.pickle"))
quantized_control_voltages = torch.load(os.path.join(path, "quantized_control_voltages.pickle"))

#creating a new model instance and setting quantized control voltages
new_model_instance = DefaultCustomModel(configs['processor'])
new_model_instance.set_control_voltages(quantized_control_voltages['fixed4'])

#finding predictions using quantized control voltages
for param in new_model_instance.parameters():
    param.requires_grad = False 
prediction_train = new_model_instance(results['train_results']['inputs'])
prediction_test = new_model_instance(results['test_results']['inputs'])

#training a perceptron to find a suitable threshold
accuracy_dict_training = get_accuracy(prediction_train, results['train_results']['targets'], configs['accuracy'], node = None)

#finding the accuracy of the model on test data
accuracy_dict_test = get_accuracy(prediction_test, results['test_results']['targets'], configs['accuracy'], node = accuracy_dict_training['node'])

#plotting perceptron 
plot_perceptron(accuracy_dict_test, save_dir=plot_dir, name="quantized")
plot_perceptron(results['test_results']['accuracy'], save_dir=plot_dir)

#plotting output of original and quantized model
plt.figure()
plt.plot(results['test_results']['best_output'], c='blue', label='original')
plt.plot(prediction_test, c='r', label='quantized')
plt.title('output (nA)')
plt.legend()
plt.savefig(os.path.join(plot_dir, "output.jpg"))

#calculating different losses for output of the model
fisherloss = loss_fn(prediction_test, results['test_results']['targets'])
difference = results['test_results']['best_output'] - prediction_test
mse = torch.sum(difference**2)/difference.size()[0]
rmse = torch.sqrt(mse)
print('mse: {}'.format(mse))
print('rmse: {}'.format(rmse))
print('fisher: {}'.format(fisherloss[0]))