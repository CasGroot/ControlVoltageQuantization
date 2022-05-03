import os
import torch
from brainspy.utils.io import load_configs
from bspytasks.models.AllInputs import AllInputs
from brainspy.utils.transforms import linear_transform

if __name__ == "__main__":
    # path to main_dir
    main_dir = r'C:\Users\CasGr\Documents\uni\BachelorOpdracht\inputspace\-1_1\NewModel'

    # getting the configs for the processor 
    configs = load_configs(r"C:\Users\CasGr\Documents\github\brainspy-tasks\configs\defaults\processors\simulation.yaml")

    # creating a processor
    model = AllInputs(configs)

    # Creating 30k uniform random values over the input space and saves them to given file
    # x = torch.FloatTensor(30000, 7).uniform_(-1, 1)
    # torch.save(x, os.path.join(main_dir, 'random_inputs.pickle'))

    # load already created inputs
    x = torch.load(os.path.join(main_dir, 'random_inputs.pickle'))
    bits=6

    # load quantized input space
    x_quant = torch.load(os.path.join(main_dir, 'quantized_input_space.pickle'))

    output_original = model(x)
    AllConfDict = {'no_electrodes': output_original}

    key = f'quantized_input_space_{bits}'
    
    x_quantized = torch.clone(x_quant[key])
    x_conf = torch.clone(x)

    # Quantize 1 electrode

    conf1 = {}

    for a in range(0, 7):
        x_conf[:, a] = x_quantized[:, a]
        output_quant = model(x_conf)
        conf1[str(a)] = output_quant
        x_conf[:, a] = x[:, a]
        AllConfDict['one_electrode'] = conf1

    # Quantize 2 electrodes

    conf2 = {}

    for a in range(0, 7):
        x_conf[:, a] = x_quantized[:, a]
        for b in range(a+1, 7):
            x_conf[:, b] = x_quantized[:, b]
            output_quant = model(x_conf)
            conf2[str(a) + str(b)] = output_quant
            x_conf[:, b] = x[:, b]
        x_conf[:, a] = x[:, a]

    AllConfDict['two_electrodes'] = conf2

    # Quantize 3 electrodes

    conf3 = {}

    for a in range(0, 7):
        x_conf[:, a] = x_quantized[:,a]
        for b in range(a+1, 7):
            x_conf[:, b] = x_quantized[:, b]
            for c in range(b+1, 7):
                x_conf[:, c] = x_quantized[:, c]
                output_quant = model(x_conf)
                conf3[str(a) + str(b) + str(c)] = output_quant
                x_conf[:, c] = x[:, c]
            x_conf[:, b] = x[:, b]
        x_conf[:, a] = x[:, a]

        AllConfDict['three_electrodes'] = conf3

    # Quantize 4 electrodes

    conf4 = {}

    for a in range(0, 7):
        x_conf[:, a] = x_quantized[:, a]
        for b in range(a+1, 7):
            x_conf[:, b] = x_quantized[:, b]
            for c in range(b+1, 7):
                x_conf[:, c] = x_quantized[:, c]
                for d in range(c+1, 7):
                    x_conf[:, d] = x_quantized[:, d]
                    output_quant = model(x_conf)
                    conf4[str(a) + str(b) + str(c) + str(d)] = output_quant
                    x_conf[:, d] = x[:, d]
                x_conf[:, c] = x[:, c]
            x_conf[:, b] = x[:, b]
        x_conf[:, a] = x[:, a]

    AllConfDict['four_electrodes'] = conf4

    # Quantize 5 electrodes

    conf5 = {}

    for a in range(0, 7):
        x_conf[:, a] = x_quantized[:, a]
        for b in range(a+1, 7):
            x_conf[:, b] = x_quantized[:, b]
            for c in range(b+1, 7):
                x_conf[:, c] = x_quantized[:, c]
                for d in range(c+1, 7):
                    x_conf[:, d] = x_quantized[:, d]
                    for e in range(d+1, 7):
                        x_conf[:, e] = x_quantized[:, e]
                        output_quant = model(x_conf)
                        conf5[str(a) + str(b) + str(c) + str(d) + str(e)] = output_quant
                        x_conf[:, e] = x[:, e]
                    x_conf[:, d] = x[:, d]
                x_conf[:, c] = x[:, c]
            x_conf[:, b] = x[:, b]
        x_conf[:, a] = x[:, a]

    AllConfDict['five_electrodes'] = conf5

    # Quantize 6 electrodes

    conf6 = {}
    x_conf = torch.clone(x_quantized)
    for a in range(6, -1, -1):
        x_conf[:, a] = x[:, a]
        output_quant = model(x_conf)
        conf6[str(a)] = output_quant
        x_conf[:, a] = x_quantized[:, a]

    AllConfDict['six_electrodes'] = conf6

    # Quantize 7 electrodes
    conf7 = {'0123456': model(x_quant[key])}

    AllConfDict['seven_electrodes'] = conf7
    torch.save(AllConfDict, os.path.join(main_dir, f'AllConfDict{bits}bits.pickle'))
