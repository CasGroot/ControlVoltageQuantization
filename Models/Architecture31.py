import torch
from torch import nn
from brainspy.processors.processor import Processor
from brainspy.processors.dnpu import DNPU
from brainspy.processors.modules.bn import DNPUBatchNorm
from brainspy.utils.pytorch import TorchUtils
#from brainspy.utils.transforms import CurrentToVoltage


class Architecture31(nn.Module):
    def __init__(self, configs, info=None, state_dict=None):
        super().__init__()
        self.alpha = 1  # configs['regul_factor']
        if info is None:
            training_data = torch.load('C:/Users/CasGr/Documents/Data/training_data_quick.pt', map_location=torch.device('cpu'))
            self.processor = Processor(
                configs,
                info=training_data['info'],
                model_state_dict=training_data['model_state_dict'])

        self.l1_nodes = 3
        self.l1_input_list = [[3, 4]] * self.l1_nodes

        self.dnpu_out = DNPU(self.processor,
                             data_input_indices=self.l1_input_list)
        self.dnpu_out.add_input_transform(input_range=[-1, 1])
        #self.dnpu_out.init_batch_norm(track_running_stats=False)
        self.linear = torch.nn.Linear(3, 1)
        
    def forward(self, x):
        x = torch.cat((x, x, x), dim=1)
        x = self.dnpu_out(x)
        x = self.linear(x)
        return x

    def format_targets(self, x):
        return self.dnpu_l1.format_targets(x)

    def hw_eval(self, configs, info=None):
        self.eval()
        self.processor.load_processor(configs, info)

    def get_input_ranges(self):
        # Necessary to implement for the automatic data input to voltage conversion
        pass

    def get_logged_variables(self):
        # Not correct
        log = {}
        dnpu_l1_logs = self.dnpu_out.get_logged_variables()
        for key in dnpu_l1_logs.keys():
            log['l1_' + key] = dnpu_l1_logs[key]

        dnpu_l2_logs = self.dnpu_l2.get_logged_variables()
        for key in dnpu_l2_logs.keys():
            log['l2_' + key] = dnpu_l2_logs[key]
        log['l3_output'] = self.output
        log['a'] = self.a
        return log

    def get_control_ranges(self):
        # Necessary to use the GA data input to voltage conversion
        control_ranges = torch.cat(
            (control_ranges, self.dnpu_out.get_control_ranges()))
        return control_ranges

    def get_control_voltages(self):
        control_voltages = torch.cat(
            (control_voltages, self.dnpu_out.get_control_voltages()))
        return control_voltages

    def set_control_voltages(self, control_voltages):
        # Needs to change
        control_voltages = control_voltages.view(5, 5)
        # Necessary to use the GA data input to voltage conversion
        self.dnpu_out.set_control_voltages(control_voltages[2])

    def get_clipping_value(self):
        return self.processor.get_clipping_value()
        # return clipping_value

    def is_hardware(self):
        return self.processor.is_hardware

    def close(self):
        self.processor.close()

    def regularizer(self):
        return self.alpha * (self.dnpu_out.regularizer())

    def constraint_weights(self):
        self.dnpu_out.constraint_control_voltages()
