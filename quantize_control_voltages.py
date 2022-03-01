import torch
from qtorch.quant import fixed_point_quantize, float_quantize


def quantize_control_voltages(filename, printvalues=True, save=True):
    control_voltages = torch.load(filename)

    float16exp5 = float_quantize(control_voltages, exp=5, man=10, rounding="nearest")
    float8exp2 = float_quantize(control_voltages, exp=2, man=5, rounding="nearest")
    fixed16frac14 = fixed_point_quantize(control_voltages, 16, 14, rounding="nearest")
    fixed8frac6 = fixed_point_quantize(control_voltages, 8, 6, rounding="nearest")
    fixed5frac3 = fixed_point_quantize(control_voltages, 5, 3, rounding="nearest")
    fixed4frac2 = fixed_point_quantize(control_voltages, 4, 2, rounding="nearest")

    quantized_control_voltages_dict = {
        'original': control_voltages,
        'float16exp5': float16exp5,
        'float8exp2': float8exp2,
        'fixed16frac14': fixed16frac14,
        'fixed8frac6': fixed8frac6,
        'fixed5frac3': fixed5frac3,
        'fixed4frac2': fixed4frac2,
    }

    if (printvalues):
        keys_list = list(quantized_control_voltages_dict)
        values_list = list(quantized_control_voltages_dict.values())
        for i in range(0, len(quantized_control_voltages_dict)):
            print(keys_list[i])
            print(values_list[i])
    if (save):
        torch.save(quantized_control_voltages_dict, 'quantized_control_voltages.pickle')


quantize_control_voltages('control_voltages.pickle')

