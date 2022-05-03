import os
import torch
from brainspy.utils.io import load_configs
# from brainspy.utils.transforms import linear_transform
from bspytasks.models.default_ring_cas import DefaultCustomModel
from bspytasks.models.Architecture21 import Architecture21
from bspytasks.models.Architecture31 import Architecture31

# path to file where plots should be saved
path = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\Results\Solutions\Architecture31\0_3_1_5_solutions_10\searcher_0.3gap_2022_04_16_010458\reproducibility'
save_dir = path

# getting the configs for the processor 
configs = load_configs(r"C:\Users\CasGr\Documents\github\brainspy-tasks\configs\defaults\processors\simulation.yaml")
training_data = torch.load(os.path.join(path, 'training_data.pickle'), map_location=torch.device('cpu')) # Either best_training_data or training_data depending on which code was used
results = torch.load(os.path.join(path, 'results.pickle'), map_location=torch.device('cpu'))

test_x = results['test_results']['inputs']
train_x = results['train_results']['inputs']
# load control voltages
quantized_control_voltages = torch.load(os.path.join(path, 'quantized_control_voltages.pickle'))

# creating a processor
model = Architecture31(configs)

model.load_state_dict(training_data['model_state_dict'])
test_output_quantized_dict = {}
train_output_quantized_dict = {}
for j in range(1, 3):
    for i in range(16, 3, -1):
        model.set_control_voltages(quantized_control_voltages['fixed' + str(i) + 'frac' + str(i-j)])
        test_output_quantized = model(test_x)
        test_output_quantized_dict[f'output{i}frac{i-j}'] = test_output_quantized
        train_output_quantized = model(train_x)
        train_output_quantized_dict[f'output{i}frac{i-j}'] = train_output_quantized
torch.save(test_output_quantized_dict, os.path.join(save_dir, 'ring_test_output_dict.pickle'))
torch.save(train_output_quantized_dict, os.path.join(save_dir, 'ring_train_output_dict.pickle'))



