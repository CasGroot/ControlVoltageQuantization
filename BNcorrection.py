import os
import torch
import torch.nn
from brainspy.utils.io import load_configs
from bspytasks.models.default_ring import DefaultCustomModel
from bspytasks.models.Architecture21 import Architecture21
from brainspy.utils.pytorch import TorchUtils
# path to reproducibility file
path = r'C:\Users\CasGr\Documents\github\brainspy-tasks\tmp\ring\ring_classification_gap_0.1_2022_03_17_160342\reproducibility'

# path to file where plots should be saved
plot_dir = 'C:/Users/CasGr/Documents/github/plots'

# loading in necessary files
configs = load_configs(os.path.join(path, "configs.yaml"))
results = torch.load(os.path.join(path, "results.pickle"))

# model_state_dict = torch.load(os.path.join(path, "model.pt"))
training_data = torch.load(os.path.join(path, "training_data.pickle"), map_location=torch.device('cpu'))

new_model_instance = TorchUtils.format(Architecture21(configs['processor']))
model = 'Arch21'

new_model_instance.load_state_dict(training_data['model_state_dict'])

quantized_control_voltages = torch.load(os.path.join(path, 'quantized_control_voltages.pickle'))

original_output = new_model_instance(results['test_results']['inputs'])

new_model_instance.set_control_voltages(quantized_control_voltages['fixed4frac2'])

quantized_output = new_model_instance(results['test_results']['inputs'])

mean = new_model_instance.dnpu_l1.get_logged_variables()['c_dnpu_output'].mean(0)
std = new_model_instance.dnpu_l1.get_logged_variables()['c_dnpu_output'].std(0)

print(new_model_instance.dnpu_l1.bn.running_mean)

new_model_instance.set_running_mean(mean)
new_model_instance.set_running_var(std**2)

print(new_model_instance.dnpu_l1.bn.running_mean)
fixedbn = new_model_instance(results['test_results']['inputs'])

loss = torch.nn.MSELoss()
print(loss(fixedbn, quantized_output))