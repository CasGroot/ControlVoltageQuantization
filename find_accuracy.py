import os
import torch
from brainspy.utils.io import load_configs
from bspytasks.models.default_ring import DefaultCustomModel
from brainspy.algorithms.modules.performance.accuracy import get_accuracy
# from brainspy.utils.manager import get_criterion
from bspytasks.ring.tasks.classifier import plot_perceptron
import matplotlib.pyplot as plt

path= 'C:/Users/CasGr/Documents/github/brainspy-tasks/tmp/ring/searcher_0.5gap_2022_02_14_200312/reproducibility/'
plot_dir= 'C:/Users/CasGr/Documents/github/plots'

configs = load_configs(os.path.join(path, "configs.yaml"))
results = torch.load(os.path.join(path, "results.pickle"))
model_state_dict = torch.load(os.path.join(path, "model.pt"))
new_model_instance = DefaultCustomModel(configs['processor'])
training_data = torch.load(os.path.join(path, "training_data.pickle"))
quantized_control_voltages = torch.load(os.path.join(path, "quantized_control_voltages.pickle"))

new_model_instance.set_control_voltages(quantized_control_voltages['fixed4'])
for param in new_model_instance.parameters():
    param.requires_grad = False 
my_result_quantized = new_model_instance(results['train_results']['inputs'])
output = new_model_instance(results['test_results']['inputs'])
# print(quantized_control_voltages)

# accuracy_dict_training = get_accuracy(my_result_quantized, results['train_results']['targets'], configs['accuracy'], node = None)
# accuracy_dict_test = get_accuracy(output, results['test_results']['targets'], configs['accuracy'], node = accuracy_dict_training['node'])

# plot_perceptron(accuracy_dict_test, save_dir=plot_dir, name="quantized")
# plot_perceptron(results['test_results']['accuracy'], save_dir=plot_dir)

# plt.figure()
# plt.plot(results['test_results']['best_output'], c='blue', label='original')
# plt.plot(output, c='r', label='quantized')
# plt.title('output (nA)')
# plt.legend()
# plt.savefig(os.path.join(plot_dir, "output.jpg"))

difference = results['test_results']['best_output'] - output
mse = torch.sum(difference**2)/difference.size()[0]
rmse = torch.sqrt(mse)
print('mse: {}'.format(mse))
print('rmse: {}'.format(rmse))