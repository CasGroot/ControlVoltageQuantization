import torch
import qtorch
from qtorch.quant import fixed_point_quantize, float_quantize

def quantize_control_voltages(filename, printvalues = True, save = True):
    control_voltages = torch.load(filename)

    float16 = float_quantize(control_voltages, exp=5, man=10, rounding="nearest")
    float8 = float_quantize(control_voltages, exp=4, man=3, rounding="nearest")
    fixed16 = fixed_point_quantize(control_voltages, 16, 14)
    fixed8 = fixed_point_quantize(control_voltages, 8, 6)
    fixed4 = fixed_point_quantize(control_voltages, 4, 2)

    quantized_control_voltages_dict = {
        'original': control_voltages,
        'float16': float16,
        'float8': float8,
        'fixed16': fixed16,
        'fixed8': fixed8,
        'fixed4': fixed4,
    }

    if (printvalues):
        keys_list = list(quantized_control_voltages_dict)
        values_list = list(quantized_control_voltages_dict.values())
        for i in range(0, len(quantized_control_voltages_dict)):
            print(keys_list[i])
            print(values_list[i])
    if (save):
        torch.save(quantized_control_voltages_dict, 'quantized_control_voltages.pickle')

quantize_control_voltages('control_voltages.pickle', True, True)