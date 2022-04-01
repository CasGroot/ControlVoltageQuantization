import os
import torch
import torch.nn
from brainspy.utils.io import load_configs
from bspytasks.models.default_ring import DefaultCustomModel
from bspytasks.models.Architecture21 import Architecture21
from brainspy.utils.pytorch import TorchUtils
import matplotlib.pyplot as plt

# path to reproducibility file
path = r'C:\Users\CasGr\Documents\github\brainspy-tasks\tmp\ring\ring_classification_gap_0.3_2022_03_21_143304_single_8nA_error\reproducibility'

# path to file where plots should be saved
plot_dir = 'C:/Users/CasGr/Documents/github/plots'

# loading in necessary files
configs = load_configs(os.path.join(path, "configs.yaml"))
results = torch.load(os.path.join(path, "results.pickle"))

# model_state_dict = torch.load(os.path.join(path, "model.pt"))
training_data = torch.load(os.path.join(path, "training_data.pickle"), map_location=torch.device('cpu'))

# create new model instance
new_model_instance = TorchUtils.format(Architecture21(configs['processor']))

# load state dict for control voltages
new_model_instance.load_state_dict(training_data['model_state_dict'])

new_model_instance.eval()

# load quantized control voltages
quantized_control_voltages = torch.load(os.path.join(path, 'quantized_control_voltages.pickle'))

# find unquantized output
original_output = new_model_instance(results['test_results']['inputs'])

# setting quantized control voltages
new_model_instance.set_control_voltages(quantized_control_voltages['fixed6frac4'])

# finding mean and variance before batch norm
new_model_instance(results['train_results']['inputs'])
mean = new_model_instance.dnpu_l1.get_logged_variables()['c_dnpu_output'].mean(0)
var = new_model_instance.dnpu_l1.get_logged_variables()['c_dnpu_output'].var(0)

# finding quantized results with original running_mean and running_var
quantized_output = new_model_instance(results['test_results']['inputs'])

print(new_model_instance.dnpu_l1.bn.running_mean)

# changing running_mean and running_var to reflect actual mean and var of quantized model more
new_model_instance.set_running_mean(mean)
new_model_instance.set_running_var(var)

print(new_model_instance.dnpu_l1.bn.running_mean)

# find quantized output with corrected batch norm
fixedbn = new_model_instance(results['test_results']['inputs'])

# plot outputs
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(original_output.detach().numpy())
ax1.set_title('original')
ax2.plot(quantized_output.detach().numpy())
ax2.set_title('quantized')
ax3.plot(fixedbn.detach().numpy())
ax3.set_title('running_mean corrected')
fig.show()
fig.savefig(os.path.join(plot_dir, "test.jpg"))
loss = torch.nn.MSELoss()
print(loss(fixedbn, quantized_output))
