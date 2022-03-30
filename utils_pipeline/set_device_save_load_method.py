from typing import Callable
import socket
import torch
from torch.cuda import Device


def get_device() -> Device:
    """
    Set the base device
    :return: device: device from where to save/load/treat data
    """
    if not torch.cuda.is_available() or socket.gethostname() == 'zh-ws1':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    return device


def set_save_load_procedure(type_files: str = 'torch') -> (Callable, Callable, str):
    """
    Set the base functions to save, load files  and define the associated extensions
    :param type_files: type of files (pickle or torch)
    :return:
        - save - function to save files
        - load - function to load files
        - format_files - extension of the files (.torch or .pickle)
    """
    if type_files == 'pickle':
        import pickle
        format_files = '.pickle'
        save = pickle.dump
        load = pickle.load
    elif type_files == 'torch':
        import torch
        format_files = '.torch'
        save = torch.save

        device = get_device()

        def load(file):
            if str(device) == 'cpu':
                loaded = torch.load(file, map_location=torch.device('cpu'))
            else:
                loaded = torch.load(file, map_location=torch.device('cuda:0'))
            return loaded

    else:
        raise NotImplementedError
    return save, load, format_files
