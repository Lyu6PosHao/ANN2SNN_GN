from copy import deepcopy
from spikingjelly.activation_based import neuron, surrogate,base
import torch
import torch.nn as nn
from typing import Callable
from torch.autograd import Function

# CombineNode is actually "IF neuron". CombineNode is easier to use in ANN2SNN.
class CombinedNode(neuron.BaseNode):
    def __init__(
        self,
        bias=0.0,
        v_threshold: float = 1.0,
        p0=0.5,
        v_reset=None,
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        super().__init__(
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            step_mode,
            backend,
            store_v_seq,
        )

        self.bias = bias
        self.v_threshold = v_threshold
        self.p0=p0
        self.v = v_threshold * p0
        self._memories_rv["v"] = v_threshold * p0

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike * self.v_threshold

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x + self.bias

    def extra_repr(self):
        if type(self.bias) is float:
            b = "0."
        else:
            b = "non_zero"
        return f" bias={self.bias}, p0={self.p0},v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}"

class GradFloor(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# QCFS is the reproduction of the paper "QCFS"
class QCFS(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.v_threshold = nn.Parameter(torch.tensor(float(T)))
        self.p0=torch.tensor(0.5)

    @staticmethod
    def floor_ste(x):
        return (x.floor() - x).detach() + x

    def extra_repr(self):
        return f"T={self.T}, p0={self.p0.item()},v_threshold={self.v_threshold.item()}"

    def forward(self, x):
        y = self.floor_ste(x * self.T / self.v_threshold +(self.p0))
        y = torch.clamp(y, 0, self.T) 
        return y* self.v_threshold / self.T

# Group Neuron
class GN(base.MemoryModule):
    def __init__(self, m:int=4, v_threshold: float = 1.0,
                 surrogate_function: Callable = surrogate.Sigmoid(),step_mode: str = 's'):
        super().__init__()
        self.m = m
        self.step_mode = step_mode
        self.surrogate_function = surrogate_function
        self.v_threshold = v_threshold/self.m
        self.register_memory("v", self.v_threshold*0.5)
        #threshholds of the member neurons: values are [v_threshold,2*v_threshold,3*v_threshold,...,m*v_threshold]
        self.bias=torch.arange(1,m+1,1)*self.v_threshold
    
    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    def single_step_forward(self, x):
        #shape is [m,N,C,H,W], an the initial value is 0.5*v_threshold
        self.v_float_to_tensor(x)
        #charge
        self.v=self.v+x
        #fire
        spike=self.surrogate_function(self.v - self.bias)
        #spike aggregation
        spike=torch.sum(spike,dim=0,keepdim=False)
        #reset (or lateral inhibition)
        self.v=self.jit_soft_reset(self.v,spike,self.v_threshold) 
        return spike*self.v_threshold
    
    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)
            self.len_vshape=len(self.v.shape)
            self.v=self.v.repeat(self.m,*[1]*self.len_vshape).to(x)
            self.bias=self.bias.view(-1,*[1]*self.len_vshape).to(x)

  

