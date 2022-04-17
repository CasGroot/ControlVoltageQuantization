import torch
from brainspy.processors.processor import Processor
from brainspy.utils.pytorch import TorchUtils


class AllInputs(torch.nn.Module):
    def __init__(self, configs):
        super(AllInputs, self).__init__()
        self.gamma = 1
        self.node_no = 1
        model_data = torch.load(configs['model_dir'],
                                map_location=TorchUtils.get_device())
        self.processor = Processor(configs, model_data['info'],
                                    model_data['model_state_dict'])

    def forward(self, x):
        x = self.processor(x)
        return x

    def hw_eval(self, configs, info=None):
        self.eval()
        self.processor.hw_eval(configs, info)

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