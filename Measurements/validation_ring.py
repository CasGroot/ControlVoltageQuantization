import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from bspytasks.ring.validation import validate
from bspytasks.ring.validation import load_reproducibility_results

# TODO: Add possibility to validate multiple times

if __name__ == "__main__":

    from torchvision import transforms

    from bspytasks.ring.validation import init_dirs
    from brainspy.utils.io import load_configs
    from bspytasks.utils.transforms import PointsToPlateaus
    from brainspy.algorithms.modules.signal import fisher

    main_dir = 'C:/Users/Unai/Documents/programming/examples-multiple-devices/tmp/TEST/output/ring/searcher_0.5gap_2021_05_11_170730'
    model, results = load_reproducibility_results(main_dir)
    # load the validation processor configurations
    configs = load_configs('processor.yaml')
    quantized_control_voltages = torch.load(os.path.join(main_dir, 'quantized_control_voltages.pickle'))
    results_dir = init_dirs(os.path.join(main_dir, 'validation'))
    results_dict = {}
    for i in range(16, 3, -1):
        model.set_control_voltages(quantized_control_voltages['fixed' + str(i) + 'frac' + str(i-2)])
        results = validate(model, results, configs, fisher, results_dir, show_plots=True)
        results_dict[str(i) + 'bit'] = results
    torch.save(results_dict, os.path.join(main_dir, 'hw_validation_results_quantized.pickle'))

