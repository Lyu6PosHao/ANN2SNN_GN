from math import sqrt
import torch.nn  as nn
import torch
from copy import deepcopy
from spikingjelly.activation_based import monitor, neuron, functional,surrogate
from modules import *


def get_module_by_name(parent, name):
    name_list = name.split(".")
    for item in name_list[:-1]:
        if hasattr(parent, item):
            parent = getattr(parent, item)
        else:
            return None, None
    if hasattr(parent, name_list[-1]):
        child = getattr(parent, name_list[-1])
        return parent, child
    else:
        return None, None      


def replace_relu_with_qcfs(model: nn.Module,L):
    input_monitor = monitor.InputMonitor(model, torch.nn.ReLU)
    for name in input_monitor.monitored_layers:
        parent, child = get_module_by_name(model, name)
        assert not (parent is None and child is None)
        new_child=QCFS(T=L)
        setattr(parent, name.split('.')[-1], new_child)
    input_monitor.clear_recorded_data()
    input_monitor.remove_hooks()
    return model

# replace qcfs with spiking neurons (IF Neurons or Group Neurons)
# "members" denotes the number of members in one group neuron.
def replace_qcfs_with_sn(model: nn.Module,members:int,sn_type:str):
    input_monitor = monitor.InputMonitor(model, QCFS)
    for name in input_monitor.monitored_layers:
        parent, child = get_module_by_name(model, name)
        assert not (parent is None and child is None)
        if sn_type=='gn':
            new_child = GN(m=members,v_threshold=child.v_threshold.item())
        elif sn_type=='if':
            new_child = CombinedNode(v_threshold=child.v_threshold.item())
        else:
            raise ValueError('sn_type must be pgn or cpgn or if')
        setattr(parent, name.split('.')[-1], new_child)
    input_monitor.clear_recorded_data()
    input_monitor.remove_hooks()
    return model

def replace_maxpool2d_with_avgpool2d(model: nn.Module):
    input_monitor = monitor.InputMonitor(model, torch.nn.MaxPool2d)
    for name in input_monitor.monitored_layers:
        parent, child = get_module_by_name(model, name)
        assert not (parent is None and child is None)
        new_child = torch.nn.AvgPool2d(child.kernel_size, child.stride, child.padding, child.ceil_mode)
        setattr(parent, name.split('.')[-1], new_child)
    input_monitor.clear_recorded_data()
    input_monitor.remove_hooks()
    return model


