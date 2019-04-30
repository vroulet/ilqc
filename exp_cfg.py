import os
import pathlib
import torch


torch.set_default_tensor_type(torch.DoubleTensor)

file_path = os.path.abspath(__file__)
exp_folder = os.path.dirname(file_path)

info_exp_folder = '{0}/exp//results'.format(exp_folder)
plots_folder = '{0}/exp/plots'.format(exp_folder)

try:
    pathlib.Path(info_exp_folder).mkdir(parents=True)
except OSError:
    pass
try:
    pathlib.Path(plots_folder).mkdir(parents=True)
except OSError:
    pass
