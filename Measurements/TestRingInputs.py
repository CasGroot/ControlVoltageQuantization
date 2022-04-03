import os
import torch
from brainspy.utils.io import load_configs
from brainspy.utils.transforms import linear_transform
from bspytasks.models.default_ring import DefaultCustomModel
# from bspytasks.models.Architecture21 import Architecture21
# from bspytasks.models.Architecture31 import Architecture31

# path to file where plots should be saved
path = r'C:\Users\CasGr\Documents\github\brainspy-tasks\tmp\ring\searcher_0.3gap_2022_03_21_101205_single_quick\reproducibility'
save_dir = r'C:\Users\CasGr\Documents\github\plots'

# getting the configs for the processor 
configs = load_configs(r"C:\Users\CasGr\Documents\github\brainspy-tasks\configs\defaults\processors\simulation.yaml")
training_data = torch.load(os.path.join(path, 'training_data.pickle')) # Either best_training_data or training_data depending on which code was used
results = torch.load(os.path.join(path, 'results.pickle'))

test_x = results['test_results']['inputs']
train_x = results['train_results']['inputs']
# load control voltages
quantized_control_voltages = torch.load(os.path.join(path, 'quantized_control_voltages.pickle'))

# creating a processor
model = DefaultCustomModel(configs)

model.load_state_dict(training_data['model_state_dict'])
test_output_quantized_dict = {}
train_output_quantized_dict = {}
for i in range(16, 3, -1):
    model.set_control_voltages(quantized_control_voltages['fixed' + str(i) + 'frac' + str(i-1)])
    test_output_quantized = model(test_x)
    test_output_quantized_dict['output' + str(i)] = test_output_quantized
    train_output_quantized = model(train_x)
    train_output_quantized_dict['output' + str(i)] = train_output_quantized
torch.save(test_output_quantized_dict, 'ring_test_output_dict.pickle')
torch.save(train_output_quantized_dict, 'ring_train_output_dict.pickle')



