import torch
from brainspy.processors.processor import Processor
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.transforms import get_linear_transform_constants

class AllInputs(torch.nn.Module):
    def __init__(self, configs):
        super(AllInputs, self).__init__()
        self.gamma = 1
        self.node_no = 1
        model_data = torch.load(configs['model_dir'],
                                map_location=TorchUtils.get_device())
        self.processor = Processor(configs, model_data['info'],
                                    model_data['model_state_dict'])
        self.scale, self.offset = get_linear_transform_constants(self.processor.get_voltage_ranges().T[0].T, 
                                                                self.processor.get_voltage_ranges().T[1].T, 
                                                                torch.tensor([-1]), torch.tensor([1]))

    def forward(self, x):
        x = (self.scale * x) + self.offset
        x = self.processor(x)
        return x

    def hw_eval(self, configs, info=None):
        self.eval()
        self.processor.swap(configs, info)

    def get_clipping_value(self):
        return self.processor.get_clipping_value()

    def is_hardware(self):
        return self.processor.processor.is_hardware

    def close(self):
        self.processor.close()

    def constraint_control_voltages(self):
        pass

    def format_targets(self, x: torch.Tensor) -> torch.Tensor:
        return self.processor.format_targets(x)